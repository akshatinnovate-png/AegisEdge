"""Diversity-aware result selection.

Five near-identical memories are one answer wearing five hats. Relevance
ranking produces exactly that, because the things most similar to the query
are also most similar to each other — and on a device whose whole job is
recalling *what happened*, a result set that hides four distinct incidents
behind one repeated phrasing is a failure mode, not a preference.

Maximal Marginal Relevance trades relevance against novelty. The lambda is not
a constant here: it adapts to how redundant the candidate set actually is, so
a genuinely varied set is not diversified for its own sake.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DiversityReport:
    selected: list[str] = field(default_factory=list)
    lambda_used: float = 0.7
    redundancy_before: float = 0.0
    redundancy_after: float = 0.0
    displaced: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"lambda": round(self.lambda_used, 3),
                "redundancy_before": round(self.redundancy_before, 4),
                "redundancy_after": round(self.redundancy_after, 4),
                "displaced": self.displaced,
                "improvement": round(self.redundancy_before - self.redundancy_after, 4)}


class MaximalMarginalRelevance:
    def __init__(self, base_lambda: float = 0.72) -> None:
        self.base_lambda = base_lambda
        self.applications = 0
        self.total_improvement = 0.0

    @staticmethod
    def _redundancy(vectors: np.ndarray) -> float:
        """Mean pairwise similarity — how much the set repeats itself."""
        if len(vectors) < 2:
            return 0.0
        sims = vectors @ vectors.T
        upper = sims[np.triu_indices(len(vectors), k=1)]
        return float(np.mean(upper))

    def select(self, query: np.ndarray, candidates: list[tuple[str, float, np.ndarray]],
               k: int) -> tuple[list[tuple[str, float]], DiversityReport]:
        if not candidates:
            return [], DiversityReport()
        self.applications += 1
        matrix = np.vstack([c[2] for c in candidates]).astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms
        relevance = np.array([c[1] for c in candidates], dtype=np.float32)

        head = matrix[: min(k, len(matrix))]
        redundancy_before = self._redundancy(head)
        # A set that is already varied needs no diversification; a set that is
        # 90% self-similar needs a lot. Adapt rather than impose.
        lam = float(np.clip(self.base_lambda - 0.35 * max(0.0, redundancy_before - 0.3), 0.3, 0.95))

        selected: list[int] = []
        remaining = list(range(len(candidates)))
        while remaining and len(selected) < k:
            if not selected:
                best = int(max(remaining, key=lambda i: relevance[i]))
            else:
                chosen = matrix[selected]
                scores = {
                    i: lam * float(relevance[i]) - (1 - lam) * float((matrix[i] @ chosen.T).max())
                    for i in remaining
                }
                best = max(scores, key=scores.get)
            selected.append(best)
            remaining.remove(best)

        report = DiversityReport(
            selected=[candidates[i][0] for i in selected],
            lambda_used=lam,
            redundancy_before=redundancy_before,
            redundancy_after=self._redundancy(matrix[selected]),
            displaced=[candidates[i][0] for i in range(min(k, len(candidates)))
                       if i not in selected],
        )
        self.total_improvement += max(0.0, report.redundancy_before - report.redundancy_after)
        return [(candidates[i][0], float(relevance[i])) for i in selected], report

    def snapshot(self) -> dict[str, Any]:
        return {"base_lambda": self.base_lambda, "applications": self.applications,
                "avg_redundancy_reduction": round(self.total_improvement / self.applications, 4)
                if self.applications else 0.0}
