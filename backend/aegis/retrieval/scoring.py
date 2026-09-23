"""Final ranking.

Similarity alone surfaces a confident memory from eight months ago over the
correct one from this morning. The final score blends similarity with recency
decay, access salience, confidence and a pin bonus, and every term is reported
so a ranking can be explained rather than trusted.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

from ..memory.schema import MemoryPoint


@dataclass(slots=True)
class ScoreWeights:
    similarity: float = 0.62
    recency: float = 0.18
    salience: float = 0.10
    confidence: float = 0.10
    pin_bonus: float = 0.08
    stale_penalty: float = 0.25
    superseded_penalty: float = 0.55


class Scorer:
    def __init__(self, half_life_days: float = 21.0, weights: ScoreWeights | None = None) -> None:
        self.half_life_s = half_life_days * 86400
        self.w = weights or ScoreWeights()

    def recency(self, point: MemoryPoint) -> float:
        return 0.5 ** (max(0.0, time.time() - point.created_at) / self.half_life_s)

    @staticmethod
    def salience(point: MemoryPoint) -> float:
        return min(1.0, math.log1p(point.access_count) / math.log(50))

    def score(self, point: MemoryPoint, similarity: float) -> tuple[float, dict[str, float]]:
        recency = self.recency(point)
        salience = self.salience(point)
        total = (
            self.w.similarity * similarity
            + self.w.recency * recency
            + self.w.salience * salience
            + self.w.confidence * point.confidence
        )
        if point.pinned:
            total += self.w.pin_bonus
        if point.stale:
            total -= self.w.stale_penalty
        if point.superseded_by:
            total -= self.w.superseded_penalty
        breakdown = {
            "similarity": round(similarity, 4),
            "recency": round(recency, 4),
            "salience": round(salience, 4),
            "confidence": round(point.confidence, 4),
            "pinned": float(point.pinned),
            "stale": float(point.stale),
            "superseded": float(bool(point.superseded_by)),
        }
        return round(max(0.0, total), 6), breakdown
