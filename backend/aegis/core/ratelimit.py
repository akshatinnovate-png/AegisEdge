"""Token buckets: sync bandwidth budget and fleet-wide reconnect storm control."""
from __future__ import annotations

import asyncio
import time


class TokenBucket:
    __slots__ = ("rate", "capacity", "_tokens", "_last", "waited_s")

    def __init__(self, rate: float, capacity: float | None = None) -> None:
        self.rate = rate
        self.capacity = capacity if capacity is not None else rate
        self._tokens = self.capacity
        self._last = time.monotonic()
        self.waited_s = 0.0

    def _refill(self) -> None:
        now = time.monotonic()
        self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
        self._last = now

    def try_take(self, amount: float = 1.0) -> bool:
        self._refill()
        if self._tokens >= amount:
            self._tokens -= amount
            return True
        return False

    async def take(self, amount: float = 1.0) -> float:
        """Await until `amount` tokens are available; returns seconds waited."""
        waited = 0.0
        while not self.try_take(amount):
            deficit = amount - self._tokens
            delay = max(0.005, deficit / self.rate)
            waited += delay
            self.waited_s += delay
            await asyncio.sleep(delay)
        return waited

    @property
    def level(self) -> float:
        self._refill()
        return self._tokens
