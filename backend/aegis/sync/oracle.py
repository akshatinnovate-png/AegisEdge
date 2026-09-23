"""Connectivity oracle.

Polling every 30 s means discovering the link 30 s late. The oracle keeps EWMA
estimates of RTT, jitter and loss, classifies the link continuously, and fires
callbacks on the transition itself — so reconnection work starts in
milliseconds instead of on the next tick.
"""
from __future__ import annotations

import asyncio
import time
from enum import Enum
from typing import Awaitable, Callable

from ..core.bus import EventBus
from ..core.metrics import METRICS


class LinkState(str, Enum):
    OFFLINE = "offline"
    DEGRADED = "degraded"
    METERED = "metered"
    HEALTHY = "healthy"


class ConnectivityOracle:
    ALPHA_RTT = 0.3          # smoothing on latency / jitter
    ALPHA_RECOVER = 0.6      # loss falls fast on a successful probe
    ALPHA_DECAY = 0.3        # loss rises slowly on a failed one

    def __init__(self, bus: EventBus, transport, interval_s: float = 3.0) -> None:
        self.bus = bus
        self.transport = transport
        self.interval_s = interval_s
        self.state = LinkState.OFFLINE
        self.rtt_ms = 0.0
        self.jitter_ms = 0.0
        self.loss = 1.0
        self.consecutive_ok = 0
        self.probes = 0
        self.transitions = 0
        self.last_transition = time.time()
        self.last_offline_at = time.time()
        self.reconnect_ms: float | None = None
        self._on_restore: list[Callable[[], Awaitable[None]]] = []
        self.forced_offline = False

    def on_restore(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Register work that must start the instant the link returns."""
        self._on_restore.append(callback)

    def classify(self) -> LinkState:
        if self.forced_offline or self.loss > 0.6:
            return LinkState.OFFLINE
        if self.loss > 0.15 or self.rtt_ms > 320 or self.jitter_ms > 90:
            return LinkState.DEGRADED
        if getattr(self.transport, "metered", False):
            return LinkState.METERED
        return LinkState.HEALTHY

    async def probe_once(self) -> LinkState:
        self.probes += 1
        t0 = time.perf_counter()
        ok = False
        if not self.forced_offline:
            try:
                ok = await self.transport.ping()
            except Exception:
                ok = False
        rtt = (time.perf_counter() - t0) * 1000.0
        # Asymmetric EWMA: a completed round trip is unambiguous evidence that
        # the link works, so recovery is fast; a single timeout is not proof of
        # an outage, so degradation is slow. Symmetric smoothing would make the
        # node sit in OFFLINE for several probes after the link is already back,
        # which is exactly the latency this subsystem exists to remove.
        if ok:
            previous = self.rtt_ms
            self.rtt_ms = self.ALPHA_RTT * rtt + (1 - self.ALPHA_RTT) * (self.rtt_ms or rtt)
            self.jitter_ms = self.ALPHA_RTT * abs(rtt - previous) + (1 - self.ALPHA_RTT) * self.jitter_ms
            self.loss = (1 - self.ALPHA_RECOVER) * self.loss
            self.consecutive_ok += 1
            if self.consecutive_ok >= 2:
                self.loss = min(self.loss, 0.05)
        else:
            self.consecutive_ok = 0
            self.loss = self.ALPHA_DECAY + (1 - self.ALPHA_DECAY) * self.loss
        METRICS.gauge("link.rtt_ms", round(self.rtt_ms, 2))
        METRICS.gauge("link.loss", round(self.loss, 3))
        await self._apply(self.classify())
        return self.state

    async def _apply(self, new_state: LinkState) -> None:
        if new_state is self.state:
            return
        previous, self.state = self.state, new_state
        self.transitions += 1
        self.last_transition = time.time()
        METRICS.incr("link.transitions")
        if new_state is LinkState.OFFLINE:
            self.last_offline_at = time.time()
            self.reconnect_ms = None
        restored = previous is LinkState.OFFLINE and new_state is not LinkState.OFFLINE
        if restored:
            self.reconnect_ms = round((time.time() - self.last_offline_at) * 1000, 1)
        self.bus.publish(
            "link", "state_changed",
            level="warn" if new_state is LinkState.OFFLINE else "ok",
            state=new_state.value, previous=previous.value,
            rtt_ms=round(self.rtt_ms, 1), loss=round(self.loss, 3),
            reconnect_ms=self.reconnect_ms,
            message=(f"link <b>{previous.value}</b> → <b>{new_state.value}</b>"
                     + (f" · recovered in {self.reconnect_ms} ms" if restored else "")),
        )
        if restored:
            for callback in self._on_restore:
                asyncio.create_task(callback())      # reconnect work starts now, not next tick

    async def run(self) -> None:
        while True:
            await self.probe_once()
            # probe faster while offline: the point is to notice the moment it returns
            await asyncio.sleep(self.interval_s * (0.35 if self.state is LinkState.OFFLINE else 1.0))

    def snapshot(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "rtt_ms": round(self.rtt_ms, 1),
            "jitter_ms": round(self.jitter_ms, 1),
            "loss": round(self.loss, 3),
            "probes": self.probes,
            "consecutive_ok": self.consecutive_ok,
            "transitions": self.transitions,
            "reconnect_ms": self.reconnect_ms,
            "forced_offline": self.forced_offline,
        }
