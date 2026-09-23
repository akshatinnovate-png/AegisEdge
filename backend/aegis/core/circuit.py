"""Per-endpoint circuit breaker.

A sick cloud endpoint must fail in microseconds, not in a 30 s timeout that
stalls the local query path behind it.
"""
from __future__ import annotations

import time
from enum import Enum

from .errors import CircuitOpen


class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        failure_threshold: int = 4,
        reset_after_s: float = 8.0,
        half_open_probes: int = 2,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_after_s = reset_after_s
        self.half_open_probes = half_open_probes
        self.state = BreakerState.CLOSED
        self.failures = 0
        self.trips = 0
        self._opened_at = 0.0
        self._probes = 0

    def allows(self) -> bool:
        if self.state is BreakerState.OPEN:
            if time.monotonic() - self._opened_at >= self.reset_after_s:
                self.state = BreakerState.HALF_OPEN
                self._probes = 0
                return True
            return False
        if self.state is BreakerState.HALF_OPEN:
            return self._probes < self.half_open_probes
        return True

    def guard(self) -> None:
        if not self.allows():
            raise CircuitOpen(f"circuit '{self.name}' is open")
        if self.state is BreakerState.HALF_OPEN:
            self._probes += 1

    def record_success(self) -> None:
        self.failures = 0
        self.state = BreakerState.CLOSED

    def record_failure(self) -> None:
        self.failures += 1
        if self.state is BreakerState.HALF_OPEN or self.failures >= self.failure_threshold:
            if self.state is not BreakerState.OPEN:
                self.trips += 1
            self.state = BreakerState.OPEN
            self._opened_at = time.monotonic()

    def snapshot(self) -> dict[str, object]:
        return {"name": self.name, "state": self.state.value, "failures": self.failures, "trips": self.trips}
