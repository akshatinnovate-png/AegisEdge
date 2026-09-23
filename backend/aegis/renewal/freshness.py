"""Freshness scoring.

Memory rots. A point that was true in March can be confidently wrong in
September, so freshness decays on a half-life and crosses into `stale` before
it can mislead a retrieval — flagged, not deleted, because the record of what
the node used to believe is itself evidence.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from ..memory.schema import MemoryPoint


@dataclass(slots=True)
class FreshnessReport:
    scanned: int = 0
    marked_stale: int = 0
    revived: int = 0
    expired: int = 0

    def as_dict(self) -> dict[str, int]:
        return {"scanned": self.scanned, "marked_stale": self.marked_stale,
                "revived": self.revived, "expired": self.expired}


class FreshnessEvaluator:
    def __init__(self, half_life_days: float = 21.0, stale_below: float = 0.35) -> None:
        self.half_life_s = half_life_days * 86400
        self.stale_below = stale_below

    def freshness(self, point: MemoryPoint) -> float:
        """Decay from last verification, damped by confidence, pinned never rots."""
        if point.pinned:
            return 1.0
        reference = point.payload.get("last_verified_at", point.updated_at)
        age = max(0.0, time.time() - float(reference))
        decay = 0.5 ** (age / self.half_life_s)
        return round(decay * max(point.confidence, 0.1), 6)

    def sweep(self, points: list[MemoryPoint]) -> FreshnessReport:
        report = FreshnessReport()
        for point in points:
            report.scanned += 1
            if point.expired():
                report.expired += 1
                continue
            score = self.freshness(point)
            point.payload["freshness"] = score
            if score < self.stale_below and not point.stale:
                point.stale = True
                report.marked_stale += 1
            elif score >= self.stale_below and point.stale:
                point.stale = False
                report.revived += 1
        return report
