"""Semantic query cache.

Operators ask the same question five different ways. Exact-match caching
misses all five; proximity in embedding space catches them, with a TTL and an
invalidation epoch so a write cannot serve stale answers.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class CacheEntry:
    vector: np.ndarray
    payload: dict[str, Any]
    namespace: str = "default"
    created_at: float = field(default_factory=time.time)
    hits: int = 0
    epoch: int = 0


class SemanticCache:
    def __init__(self, threshold: float = 0.97, ttl_s: float = 45.0, capacity: int = 256) -> None:
        self.threshold = threshold
        self.ttl_s = ttl_s
        self.capacity = capacity
        self.entries: list[CacheEntry] = []
        self.epoch = 0
        self.hits = 0
        self.misses = 0
        self.cross_namespace_blocks = 0

    def invalidate(self) -> None:
        """A write bumps the epoch; nothing older can be served."""
        self.epoch += 1

    def get(self, vector: np.ndarray, namespace: str = "default") -> dict[str, Any] | None:
        """Look up within one namespace only.

        A cache keyed on the query embedding alone is a cross-tenant leak
        waiting to happen: two tenants asking the same question produce the
        same vector, and the second one is served the first one's documents.
        The namespace is part of the key, not a filter applied afterwards.
        """
        now = time.time()
        self.entries = [e for e in self.entries
                        if now - e.created_at < self.ttl_s and e.epoch == self.epoch]
        candidates = [e for e in self.entries if e.namespace == namespace]
        blocked = [e for e in self.entries if e.namespace != namespace]
        if not candidates:
            self.misses += 1
            if blocked:
                self.cross_namespace_blocks += 1
            return None
        matrix = np.vstack([e.vector for e in candidates])
        sims = matrix @ vector
        best = int(np.argmax(sims))
        if float(sims[best]) >= self.threshold:
            entry = candidates[best]
            entry.hits += 1
            self.hits += 1
            return {**entry.payload, "cached": True,
                    "cache_similarity": round(float(sims[best]), 4)}
        self.misses += 1
        return None

    def put(self, vector: np.ndarray, payload: dict[str, Any], namespace: str = "default") -> None:
        self.entries.append(CacheEntry(vector=vector, payload=payload,
                                       namespace=namespace, epoch=self.epoch))
        if len(self.entries) > self.capacity:
            self.entries.sort(key=lambda e: (e.hits, e.created_at))
            del self.entries[: len(self.entries) - self.capacity]

    def snapshot(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {"entries": len(self.entries), "hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / total, 3) if total else 0.0, "epoch": self.epoch,
                "namespaces": len({e.namespace for e in self.entries}),
                "cross_namespace_blocks": self.cross_namespace_blocks}
