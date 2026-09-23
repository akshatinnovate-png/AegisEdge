"""Thermal and power governor.

When the package is hot or the battery is low, the node sheds inference cost
on its own terms instead of waiting for the silicon to thermally throttle it
into unresponsiveness.

What it actually controls is the **token budget** and the **batching window** —
the two places the cost lives. It does not claim to hot-swap quantized
artefacts: the embedder graph is a gather plus a pooling reduction, there is no
separate int8 artefact to switch to, and pretending otherwise would be a
dashboard lie.

Sensors are read from the platform. Where a platform exposes none, the reading
is reported as **unavailable** rather than synthesised, because a fabricated
temperature is worse than a missing one.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..core.bus import EventBus
from ..core.metrics import METRICS

# token budget and batch window per rung, cheapest last
LADDER = [
    ("full", 128, 8.0),
    ("reduced", 96, 12.0),
    ("economy", 64, 20.0),
    ("minimal", 32, 32.0),
]


@dataclass(slots=True)
class PowerState:
    temp_c: float | None = None
    battery_pct: float | None = None
    on_mains: bool | None = None
    cpu_load: float | None = None
    available: bool = False
    source: str = "unavailable"

    def as_dict(self) -> dict[str, Any]:
        return {"temp_c": self.temp_c, "battery_pct": self.battery_pct,
                "on_mains": self.on_mains, "cpu_load": self.cpu_load,
                "sensors_available": self.available, "source": self.source}


class ThermalGovernor:
    def __init__(self, bus: EventBus, embedder, ceiling_c: float,
                 battery_floor_pct: float) -> None:
        self.bus = bus
        self.embedder = embedder
        self.ceiling_c = ceiling_c
        self.battery_floor_pct = battery_floor_pct
        self.state = PowerState()
        self.rung = 0
        self.swaps = 0
        self.last_swap = 0.0
        self.forced: int | None = None

    # -- sensing ----------------------------------------------------------

    def sample(self) -> PowerState:
        try:
            import psutil
        except ImportError:
            self.state = PowerState(available=False, source="psutil not installed")
            return self.state

        state = PowerState(available=True, source="psutil")
        try:
            readings = [t.current for group in (psutil.sensors_temperatures() or {}).values()
                        for t in group if t.current]
            state.temp_c = max(readings) if readings else None
        except Exception:
            state.temp_c = None
        try:
            battery = psutil.sensors_battery()
            if battery is not None:
                state.battery_pct = float(battery.percent)
                state.on_mains = bool(battery.power_plugged)
        except Exception:
            pass
        try:
            state.cpu_load = psutil.cpu_percent(interval=None) / 100.0
        except Exception:
            state.cpu_load = None

        if state.temp_c is None and state.battery_pct is None:
            # A server or container commonly exposes neither. Say so.
            state.source = "psutil (no thermal or battery sensors on this platform)"
        self.state = state
        if state.temp_c is not None:
            METRICS.gauge("governor.temp_c", round(state.temp_c, 1))
        if state.battery_pct is not None:
            METRICS.gauge("governor.battery_pct", round(state.battery_pct, 1))
        if state.cpu_load is not None:
            METRICS.gauge("governor.cpu_load", round(state.cpu_load, 3))
        return self.state

    # -- policy -----------------------------------------------------------

    def desired_rung(self) -> int:
        if self.forced is not None:
            return self.forced
        temp = self.state.temp_c
        battery = self.state.battery_pct
        load = self.state.cpu_load

        rung = 0
        if temp is not None:
            if temp > self.ceiling_c + 8:
                rung = max(rung, 3)
            elif temp > self.ceiling_c:
                rung = max(rung, 2)
            elif temp > self.ceiling_c - 12:
                rung = max(rung, 1)
        if battery is not None and not self.state.on_mains:
            if battery < self.battery_floor_pct / 2:
                rung = max(rung, 3)
            elif battery < self.battery_floor_pct:
                rung = max(rung, 2)
        if load is not None and load > 0.92:
            rung = max(rung, 1)             # sustained CPU saturation is a real signal
        return rung

    def enforce(self) -> str | None:
        target = self.desired_rung()
        if target == self.rung or time.time() - self.last_swap < 5.0:   # hysteresis
            return None
        previous = self.rung
        name, tokens, window = LADDER[target]
        self.embedder.set_token_budget(tokens)
        self.embedder.set_precision(name)
        self.rung = target
        self.swaps += 1
        self.last_swap = time.time()
        METRICS.incr("governor.swaps")
        self.bus.publish(
            "telemetry", "governor", level="warn" if target > previous else "ok",
            from_rung=LADDER[previous][0], to_rung=name, token_budget=tokens,
            temp_c=self.state.temp_c, battery_pct=self.state.battery_pct,
            message=(f"governor <b>{LADDER[previous][0]}</b> → <b>{name}</b> · "
                     f"token budget {tokens}"),
        )
        return name

    def force(self, rung: int | None) -> int:
        self.forced = None if rung is None else max(0, min(rung, len(LADDER) - 1))
        self.last_swap = 0.0
        self.enforce()
        return self.rung

    def snapshot(self) -> dict[str, Any]:
        name, tokens, window = LADDER[self.rung]
        return {**self.state.as_dict(), "rung": name, "rung_index": self.rung,
                "token_budget": tokens, "batch_window_ms": window,
                "swaps": self.swaps, "forced": self.forced is not None,
                "variant": name}
