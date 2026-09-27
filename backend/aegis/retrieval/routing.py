"""Which engine answers this query.

The node has two implementations of the same hybrid pipeline: Qdrant, which
runs all four stages in one call and can explain the plan it executed, and the
local adaptive index, which is a calibrated HNSW with a quantized cold tier in
this process. The bake-off measures both and they agree stage for stage, so the
choice is not about correctness. It is about latency, and latency is only
meaningful against an objective.

So the router does not hard-code a winner. It sends queries to the engine while
the engine's own measured p95 leaves room under the query objective, and moves
to the local index when it does not — under load, under a burning error budget,
or on a machine where the engine turns out to be slower than this run assumed.
Because a path that stops being used stops being measured, it keeps sampling
the one it is not using, so the decision is revisited rather than frozen at
whatever the first few queries happened to show.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Decision:
    path: str               # "engine" or "index"
    reason: str
    sampled: bool = False   # taken to keep the unused path's estimate live

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "reason": self.reason, "sampled": self.sampled}


class PathRouter:
    """Latency-aware routing between the engine and the local index.

    `policy` is "auto", "engine" or "index"; the last two are forcing, and
    exist so the bake-off can measure a path the router would not have picked.
    """

    # How much of the objective a path may consume before it is not worth it.
    # A quarter is deliberate: retrieval is one stage of a request that also
    # embeds, reranks, scores and serialises, so spending the whole objective
    # on recall would meet the number and miss the point.
    SHARE_OF_OBJECTIVE = 0.25
    SAMPLE_EVERY = 32          # keep the unused path's estimate from going stale
    WARMUP = 4                 # queries before the engine's p95 means anything

    def __init__(self, policy: str = "auto", objective_ms: float = 150.0,
                 window: int = 64) -> None:
        self.policy = policy
        self.objective_ms = objective_ms
        self.engine_ms: deque[float] = deque(maxlen=window)
        self.index_ms: deque[float] = deque(maxlen=window)
        self.queries = 0
        self.to_engine = 0
        self.to_index = 0
        self.samples = 0
        self.last: Decision | None = None

    @staticmethod
    def _p95(samples: deque[float]) -> float:
        if not samples:
            return 0.0
        ordered = sorted(samples)
        return ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))]

    @property
    def budget_ms(self) -> float:
        return self.objective_ms * self.SHARE_OF_OBJECTIVE

    def observe(self, path: str, ms: float) -> None:
        (self.engine_ms if path == "engine" else self.index_ms).append(float(ms))

    def choose(self, *, engine_available: bool, server: bool, degraded: bool) -> Decision:
        """Pick a path, and say why in words an operator can act on."""
        self.queries += 1
        decision = self._choose(engine_available=engine_available, server=server,
                                degraded=degraded)
        self.last = decision
        if decision.path == "engine":
            self.to_engine += 1
        else:
            self.to_index += 1
        if decision.sampled:
            self.samples += 1
        return decision

    def _choose(self, *, engine_available: bool, server: bool, degraded: bool) -> Decision:
        if not engine_available:
            return Decision("index", "no Qdrant-native path on this store")
        if self.policy == "index":
            return Decision("index", "policy pins queries to the local index")
        if self.policy == "engine":
            return Decision("engine", "policy pins queries to the engine")
        if server:
            # A Qdrant Server keeps the postings warm in a compiled engine and
            # the single call removes four round trips; there is nothing to weigh.
            return Decision("engine", "qdrant server: one call beats four")
        if degraded:
            # Shedding work is the SLO ladder's job. Taking the slower of two
            # paths while it sheds features would undo what it is defending.
            if self.queries % self.SAMPLE_EVERY == 0:
                return Decision("engine", "sampling the engine while degraded", sampled=True)
            return Decision("index", "degraded: the faster path defends the budget")

        engine = self._p95(self.engine_ms)
        if len(self.engine_ms) < self.WARMUP:
            return Decision("engine", f"measuring the engine ({len(self.engine_ms)} "
                                      f"of {self.WARMUP} samples)")
        if engine <= self.budget_ms:
            return Decision("engine", f"engine p95 {engine:.1f} ms is within the "
                                      f"{self.budget_ms:.0f} ms recall budget")
        index = self._p95(self.index_ms)
        if index and index >= engine:
            return Decision("engine", f"engine p95 {engine:.1f} ms is over budget but the "
                                      f"local index is no better ({index:.1f} ms)")
        if self.queries % self.SAMPLE_EVERY == 0:
            return Decision("engine", f"sampling the engine ({engine:.1f} ms p95) to keep "
                                      f"the comparison live", sampled=True)
        return Decision("index", f"engine p95 {engine:.1f} ms exceeds the "
                                 f"{self.budget_ms:.0f} ms recall budget")

    def snapshot(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "objective_ms": self.objective_ms,
            "recall_budget_ms": round(self.budget_ms, 1),
            "engine_p95_ms": round(self._p95(self.engine_ms), 3),
            "index_p95_ms": round(self._p95(self.index_ms), 3),
            "queries": self.queries, "to_engine": self.to_engine,
            "to_index": self.to_index, "samples": self.samples,
            "last": self.last.as_dict() if self.last else None,
        }
