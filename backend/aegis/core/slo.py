"""Service-level objectives and the degradation ladder.

Under load, most systems degrade by getting slower until something times out —
which fails every request instead of protecting most of them. A node that has
to survive a thermal event, a burst of ingest and a reconnect storm at the
same time needs to choose *what to stop doing*, deliberately, before the
latency budget is gone.

So the node runs an explicit ladder. Each rung sheds the most expensive stage
that is still enabled, cheapest-value-first, and the decision is driven by a
burn-rate policy on an error budget rather than by an instantaneous spike:

    0  FULL         everything on
    1  ECONOMISE    no cloud escalation; smaller candidate fetch
    2  TRIM         no late interaction, no adapter rescoring
    3  ESSENTIAL    no cross-encoder rerank; dense+sparse fusion only
    4  SURVIVAL     dense only, minimal k, background work suspended

Rungs are entered on sustained burn and left with hysteresis, so the node does
not oscillate. Every transition names what was disabled and why, because a
silently degraded system is indistinguishable from a broken one.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from .bus import EventBus
from .metrics import METRICS


class Level(IntEnum):
    FULL = 0
    ECONOMISE = 1
    TRIM = 2
    ESSENTIAL = 3
    SURVIVAL = 4


# What each rung switches off. A feature is available if its first rung is
# strictly above the current level.
FEATURES: dict[str, Level] = {
    "triton_escalation": Level.ECONOMISE,
    "wide_fetch": Level.ECONOMISE,
    "graph_boost": Level.ECONOMISE,
    "late_interaction": Level.TRIM,
    "adapter": Level.TRIM,
    "query_understanding": Level.TRIM,
    "cross_encoder": Level.ESSENTIAL,
    "diversity": Level.ESSENTIAL,
    "sparse": Level.SURVIVAL,
    "background_maintenance": Level.SURVIVAL,
}

RUNG_REASON = {
    Level.FULL: "all stages enabled",
    Level.ECONOMISE: "cloud escalation and wide fetches suspended",
    Level.TRIM: "late interaction, adapter and query rewriting suspended",
    Level.ESSENTIAL: "cross-encoder rerank and diversity suspended — fusion only",
    Level.SURVIVAL: "dense retrieval only; background work suspended",
}


@dataclass
class Objective:
    name: str
    target_ms: float | None = None
    target_success: float | None = None
    window_s: float = 300.0

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "target_ms": self.target_ms,
                "target_success": self.target_success, "window_s": self.window_s}


@dataclass
class Transition:
    at: float
    frm: Level
    to: Level
    reason: str
    burn_rate: float

    def as_dict(self) -> dict[str, Any]:
        return {"at": self.at, "from": self.frm.name, "to": self.to.name,
                "reason": self.reason, "burn_rate": round(self.burn_rate, 3)}


class SLOManager:
    """Error-budget accounting plus the ladder that defends it."""

    # burn-rate thresholds, in multiples of the budget's sustainable rate
    ESCALATE_BURN = 2.0
    DEESCALATE_BURN = 0.5
    MIN_DWELL_S = 8.0          # hysteresis: do not flap between rungs
    MIN_SAMPLES = 12

    def __init__(self, bus: EventBus, latency_target_ms: float = 150.0,
                 success_target: float = 0.995, window: int = 512) -> None:
        self.bus = bus
        self.objective = Objective("query", latency_target_ms, success_target)
        self.latencies: deque[float] = deque(maxlen=window)
        self.outcomes: deque[bool] = deque(maxlen=window)
        self.level = Level.FULL
        self.transitions: list[Transition] = []
        self.entered_at = time.time()
        self.manual_override: Level | None = None
        self.budget_consumed = 0.0
        self.shed_events = 0

    # -- measurement ------------------------------------------------------

    def observe(self, latency_ms: float, ok: bool = True) -> None:
        self.latencies.append(float(latency_ms))
        self.outcomes.append(bool(ok))
        if not ok or latency_ms > (self.objective.target_ms or float("inf")):
            self.budget_consumed += 1.0
        METRICS.gauge("slo.level", int(self.level))

    def percentile(self, q: float) -> float:
        if not self.latencies:
            return 0.0
        ordered = sorted(self.latencies)
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    @property
    def success_rate(self) -> float:
        return sum(self.outcomes) / len(self.outcomes) if self.outcomes else 1.0

    @property
    def burn_rate(self) -> float:
        """How fast the budget is being spent, relative to sustainable."""
        if len(self.latencies) < self.MIN_SAMPLES:
            return 0.0
        target = self.objective.target_ms or float("inf")
        breaches = sum(1 for value in self.latencies if value > target)
        failures = sum(1 for ok in self.outcomes if not ok)
        observed = (breaches + failures) / max(len(self.latencies), 1)
        allowed = max(1.0 - (self.objective.target_success or 0.995), 1e-4)
        return observed / allowed

    # -- the ladder -------------------------------------------------------

    def allows(self, feature: str) -> bool:
        """The single question every optional stage asks before running."""
        rung = FEATURES.get(feature)
        if rung is None:
            return True
        return self.level < rung

    def disabled(self) -> list[str]:
        return sorted(name for name, rung in FEATURES.items() if self.level >= rung)

    def evaluate(self, pressure: float = 0.0) -> Level:
        """Decide the rung from burn rate and scheduler pressure."""
        if self.manual_override is not None:
            return self._transition(self.manual_override, "manual override", self.burn_rate)
        if time.time() - self.entered_at < self.MIN_DWELL_S:
            return self.level

        burn = self.burn_rate
        if burn >= self.ESCALATE_BURN or pressure > 0.9:
            target = Level(min(int(self.level) + 1, int(Level.SURVIVAL)))
            reason = (f"burn rate {burn:.1f}x over budget" if burn >= self.ESCALATE_BURN
                      else f"scheduler pressure {pressure:.0%}")
            return self._transition(target, reason, burn)
        if burn <= self.DEESCALATE_BURN and pressure < 0.5 and self.level > Level.FULL:
            target = Level(int(self.level) - 1)
            return self._transition(target, f"burn rate {burn:.2f}x — recovering", burn)
        return self.level

    def _transition(self, target: Level, reason: str, burn: float) -> Level:
        if target == self.level:
            return self.level
        previous, self.level = self.level, target
        self.entered_at = time.time()
        record = Transition(time.time(), previous, target, reason, burn)
        self.transitions.append(record)
        if target > previous:
            self.shed_events += 1
        METRICS.incr(f"slo.transition.{target.name.lower()}")
        self.bus.publish(
            "alerts", "degradation",
            level="error" if target >= Level.ESSENTIAL else "warn" if target > Level.FULL else "ok",
            **record.as_dict(), disabled=self.disabled(),
            message=(f"degradation <b>{previous.name}</b> → <b>{target.name}</b> · {reason} · "
                     f"{RUNG_REASON[target]}"),
        )
        return self.level

    def override(self, level: Level | None) -> Level:
        self.manual_override = level
        if level is not None:
            self._transition(level, "manual override", self.burn_rate)
        else:
            # Releasing a pin hands control back to the evidence. With no
            # evidence of trouble there is no reason to stay degraded — walking
            # down one rung per tick would leave the node quietly crippled long
            # after the operator released the pin.
            self.entered_at = 0.0
            if len(self.latencies) < self.MIN_SAMPLES or self.burn_rate <= self.DEESCALATE_BURN:
                self._transition(Level.FULL, "override released — evidence does not justify shedding",
                                 self.burn_rate)
            else:
                self.evaluate()
        return self.level

    # -- reporting --------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        return {
            "level": self.level.name, "level_index": int(self.level),
            "reason": RUNG_REASON[self.level],
            "disabled_features": self.disabled(),
            "objective": self.objective.as_dict(),
            "p50_ms": round(self.percentile(0.5), 2),
            "p95_ms": round(self.percentile(0.95), 2),
            "p99_ms": round(self.percentile(0.99), 2),
            "success_rate": round(self.success_rate, 5),
            "burn_rate": round(self.burn_rate, 3),
            "budget_consumed": round(self.budget_consumed, 1),
            "samples": len(self.latencies),
            "shed_events": self.shed_events,
            "manual_override": self.manual_override.name if self.manual_override else None,
            "recent_transitions": [t.as_dict() for t in self.transitions[-5:]],
        }
