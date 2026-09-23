"""Rank fusion.

Dense and sparse scores are not on the same scale, so they cannot be added.
Reciprocal Rank Fusion combines the two *orderings* instead, which is robust
without any per-corpus tuning — the right property for a device that has no
one to tune it.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class FusedHit:
    point_id: str
    rrf: float = 0.0
    dense_rank: int | None = None
    sparse_rank: int | None = None
    dense_score: float = 0.0
    sparse_score: float = 0.0
    contributors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {"id": self.point_id, "rrf": round(self.rrf, 6),
                "dense_rank": self.dense_rank, "sparse_rank": self.sparse_rank,
                "dense_score": round(self.dense_score, 4),
                "sparse_score": round(self.sparse_score, 4),
                "matched_by": self.contributors}


def reciprocal_rank_fusion(
    dense: list[tuple[str, float]],
    sparse: list[tuple[str, float]],
    k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 0.85,
) -> list[FusedHit]:
    hits: dict[str, FusedHit] = {}
    for rank, (point_id, score) in enumerate(dense, start=1):
        hit = hits.setdefault(point_id, FusedHit(point_id))
        hit.dense_rank, hit.dense_score = rank, score
        hit.rrf += dense_weight / (k + rank)
        hit.contributors.append("dense")
    for rank, (point_id, score) in enumerate(sparse, start=1):
        hit = hits.setdefault(point_id, FusedHit(point_id))
        hit.sparse_rank, hit.sparse_score = rank, score
        hit.rrf += sparse_weight / (k + rank)
        hit.contributors.append("sparse")
    return sorted(hits.values(), key=lambda h: -h.rrf)
