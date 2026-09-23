"""The compactor.

Placement is an access-recency x salience decision, not a fixed rule: hot
memories stay in full precision, the long tail sinks to int8 and then to 1 bit.
Pinned points never sink. Expired points leave the resident set entirely.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

from .schema import MemoryPoint, Tier


@dataclass(slots=True)
class CompactionReport:
    promoted: int = 0
    demoted: int = 0
    evicted: int = 0
    scanned: int = 0
    duration_ms: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "promoted": self.promoted, "demoted": self.demoted,
            "evicted": self.evicted, "scanned": self.scanned,
            "duration_ms": round(self.duration_ms, 2),
        }


class TieringPolicy:
    def __init__(self, hot_capacity: int, warm_capacity: int, half_life_s: float = 6 * 3600) -> None:
        self.hot_capacity = hot_capacity
        self.warm_capacity = warm_capacity
        self.half_life_s = half_life_s

    def salience(self, point: MemoryPoint) -> float:
        """Recency decay x log access frequency x confidence, pinned dominates."""
        if point.pinned:
            return math.inf
        idle = max(0.0, time.time() - point.last_access_at)
        recency = 0.5 ** (idle / self.half_life_s)
        frequency = math.log1p(point.access_count)
        return recency * (1.0 + frequency) * max(point.confidence, 0.05)

    def plan(self, points: list[MemoryPoint]) -> dict[str, Tier]:
        """Rank every resident point once and cut the ranking into tiers."""
        ranked = sorted(points, key=self.salience, reverse=True)
        plan: dict[str, Tier] = {}
        for rank, point in enumerate(ranked):
            if point.expired():
                plan[point.id] = Tier.EVICTED
            elif rank < self.hot_capacity:
                plan[point.id] = Tier.HOT
            elif rank < self.hot_capacity + self.warm_capacity:
                plan[point.id] = Tier.WARM
            else:
                plan[point.id] = Tier.COLD
        return plan
