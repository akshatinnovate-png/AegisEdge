"""Decorrelated jitter backoff.

Exponential backoff synchronises a fleet: every device retries on the same
beat and the coordinator is hit by a thundering herd. Decorrelated jitter
(AWS's formulation) spreads them out.
"""
from __future__ import annotations

import random


class DecorrelatedJitter:
    __slots__ = ("base_ms", "cap_ms", "_sleep_ms", "attempts")

    def __init__(self, base_ms: float = 120.0, cap_ms: float = 20_000.0) -> None:
        self.base_ms = base_ms
        self.cap_ms = cap_ms
        self._sleep_ms = base_ms
        self.attempts = 0

    def next_delay_ms(self) -> float:
        self.attempts += 1
        self._sleep_ms = min(self.cap_ms, random.uniform(self.base_ms, self._sleep_ms * 3.0))
        return round(self._sleep_ms, 2)

    def reset(self) -> None:
        self._sleep_ms = self.base_ms
        self.attempts = 0
