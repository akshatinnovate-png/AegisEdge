"""Tiered vector index: full-precision HOT, int8 WARM, binary COLD.

Search fans out across all three tiers, over-fetching from the lossy tiers and
rescoring the survivors against full precision, then merges. A sparse inverted
index runs alongside so rare-token lexical matches survive offline, where
there is no cloud reranker to rescue recall.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from .quantize import BinaryQuantizer, ScalarQuantizer
from .schema import Tier


class _TierBlock:
    """Column store for one tier: ids + whatever encoding that tier uses."""

    def __init__(self, dim: int, tier: Tier) -> None:
        self.dim = dim
        self.tier = tier
        self.ids: list[str] = []
        self.pos: dict[str, int] = {}
        self.full = np.zeros((0, dim), dtype=np.float32)   # kept for rescoring
        self.codes: np.ndarray | None = None
        self.scales: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.ids)

    def add(self, point_id: str, vector: np.ndarray) -> None:
        if point_id in self.pos:
            self.update(point_id, vector)
            return
        self.pos[point_id] = len(self.ids)
        self.ids.append(point_id)
        self.full = np.vstack([self.full, vector.reshape(1, -1)]) if len(self.full) else vector.reshape(1, -1).copy()
        self._reencode()

    def update(self, point_id: str, vector: np.ndarray) -> None:
        idx = self.pos.get(point_id)
        if idx is None:
            return
        self.full[idx] = vector
        self._reencode()

    def remove(self, point_id: str) -> None:
        idx = self.pos.pop(point_id, None)
        if idx is None:
            return
        self.ids.pop(idx)
        self.full = np.delete(self.full, idx, axis=0)
        for pid, p in self.pos.items():
            if p > idx:
                self.pos[pid] = p - 1
        self._reencode()

    def _reencode(self) -> None:
        if self.tier is Tier.WARM and len(self.full):
            self.codes, self.scales = ScalarQuantizer.encode(self.full)
        elif self.tier is Tier.COLD and len(self.full):
            self.codes = BinaryQuantizer.encode(self.full)
        else:
            self.codes = self.scales = None

    def search(self, query: np.ndarray, k: int, oversample: int) -> list[tuple[str, float, bool]]:
        """Returns (id, score, rescored). Lossy tiers over-fetch then rescore."""
        if not len(self.ids):
            return []
        if self.tier is Tier.HOT or self.codes is None:
            scores = self.full @ query
            order = np.argsort(-scores)[:k]
            return [(self.ids[i], float(scores[i]), False) for i in order]

        fetch = min(len(self.ids), k * oversample)
        if self.tier is Tier.WARM:
            approx = ScalarQuantizer.decode(self.codes, self.scales) @ query
        else:
            approx = BinaryQuantizer.similarity(
                np.packbits(query > 0).reshape(1, -1), self.codes, self.dim
            )
        cand = np.argsort(-approx)[:fetch]
        exact = self.full[cand] @ query                      # rescore survivors
        order = np.argsort(-exact)[:k]
        return [(self.ids[cand[i]], float(exact[i]), True) for i in order]


class SparseIndex:
    """Inverted index over hashed term ids with impact-ordered postings."""

    def __init__(self) -> None:
        self.postings: dict[int, dict[str, float]] = defaultdict(dict)
        self.norms: dict[str, float] = {}

    def add(self, point_id: str, sparse: dict[int, float]) -> None:
        self.remove(point_id)
        norm = 0.0
        for term, weight in sparse.items():
            self.postings[term][point_id] = weight
            norm += weight * weight
        self.norms[point_id] = norm ** 0.5 or 1.0

    def remove(self, point_id: str) -> None:
        if point_id not in self.norms:
            return
        for term in list(self.postings):
            self.postings[term].pop(point_id, None)
            if not self.postings[term]:
                del self.postings[term]
        self.norms.pop(point_id, None)

    def search(self, sparse: dict[int, float], k: int) -> list[tuple[str, float]]:
        if not sparse:
            return []
        acc: dict[str, float] = defaultdict(float)
        qnorm = (sum(w * w for w in sparse.values()) ** 0.5) or 1.0
        for term, qw in sparse.items():
            for pid, w in self.postings.get(term, {}).items():
                acc[pid] += qw * w
        scored = [(pid, s / (qnorm * self.norms.get(pid, 1.0))) for pid, s in acc.items()]
        scored.sort(key=lambda x: -x[1])
        return scored[:k]


class TieredIndex:
    """The dense side of one collection, spread across three tiers."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.blocks = {t: _TierBlock(dim, t) for t in (Tier.HOT, Tier.WARM, Tier.COLD)}
        self.sparse = SparseIndex()
        self.tier_of: dict[str, Tier] = {}
        self.rescored = 0

    def upsert(self, point_id: str, dense: np.ndarray, sparse: dict[int, float], tier: Tier) -> None:
        current = self.tier_of.get(point_id)
        if current is not None and current is not tier:
            self.blocks[current].remove(point_id)
        self.blocks[tier].add(point_id, dense)
        self.tier_of[point_id] = tier
        self.sparse.add(point_id, sparse)

    def move(self, point_id: str, tier: Tier) -> bool:
        current = self.tier_of.get(point_id)
        if current is None or current is tier:
            return False
        block = self.blocks[current]
        idx = block.pos.get(point_id)
        if idx is None:
            return False
        vector = block.full[idx].copy()
        block.remove(point_id)
        self.blocks[tier].add(point_id, vector)
        self.tier_of[point_id] = tier
        return True

    def remove(self, point_id: str) -> None:
        tier = self.tier_of.pop(point_id, None)
        if tier is not None:
            self.blocks[tier].remove(point_id)
        self.sparse.remove(point_id)

    def search_dense(self, query: np.ndarray, k: int, oversample: int = 4) -> list[tuple[str, float]]:
        merged: list[tuple[str, float, bool]] = []
        for block in self.blocks.values():
            merged.extend(block.search(query, k, oversample))
        self.rescored += sum(1 for _, _, r in merged if r)
        merged.sort(key=lambda x: -x[1])
        return [(pid, score) for pid, score, _ in merged[:k]]

    def search_sparse(self, sparse: dict[int, float], k: int) -> list[tuple[str, float]]:
        return self.sparse.search(sparse, k)

    def counts(self) -> dict[str, int]:
        return {t.value: len(b) for t, b in self.blocks.items()}

    def __len__(self) -> int:
        return sum(len(b) for b in self.blocks.values())

    def ids(self) -> Iterable[str]:
        return self.tier_of.keys()
