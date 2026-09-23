"""Adaptive ANN index.

There is no single best index. Flat BLAS beats a graph on small collections
because 4000x96 floats fit in L2 and NumPy runs at memory bandwidth, while a
graph walk pays Python-level indirection per hop. HNSW wins once the corpus
outgrows cache. IVF-PQ wins when the corpus outgrows RAM.

So the node does not pick one at build time. It microbenchmarks the *actual
device* at boot, derives the crossover points, and each collection is placed
on the strategy its size and access pattern justify — re-evaluated as it grows.
That calibration is reported, not assumed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .hnsw import HnswIndex, HnswParams
from .pq import IvfPqIndex, PqParams


class Strategy(str, Enum):
    FLAT = "flat"          # exact, BLAS
    HNSW = "hnsw"          # graph, approximate
    IVF_PQ = "ivf_pq"      # compressed, approximate, rescored


@dataclass
class CostModel:
    """Measured, not hard-coded."""
    flat_ns_per_point: float = 0.0
    graph_ns_per_hop: float = 0.0
    hnsw_crossover: int = 50_000
    ivf_crossover: int = 500_000
    calibrated: bool = False
    detail: dict[str, float] = field(default_factory=dict)

    def calibrate(self, dim: int = 384, sample: int = 4096) -> "CostModel":
        rng = np.random.default_rng(0)
        data = rng.normal(size=(sample, dim)).astype(np.float32)
        data /= np.linalg.norm(data, axis=1, keepdims=True)
        query = data[0]

        t0 = time.perf_counter()
        for _ in range(5):
            np.argsort(-(data @ query))[:10]
        flat_ns = (time.perf_counter() - t0) / (5 * sample) * 1e9

        # a graph hop is a small gather plus a dot product over a neighbour set
        neighbours = rng.integers(0, sample, size=32)
        t0 = time.perf_counter()
        for _ in range(500):
            data[neighbours] @ query
        hop_ns = (time.perf_counter() - t0) / (500 * 32) * 1e9

        self.flat_ns_per_point = flat_ns
        self.graph_ns_per_hop = hop_ns
        # HNSW touches ~ef*log(n) points; it pays off once flat's linear scan
        # costs more than that bounded walk, including the interpreter overhead
        # the measurement above already contains.
        overhead = max(hop_ns / max(flat_ns, 1e-9), 1.0)
        self.hnsw_crossover = int(max(5_000, 64 * overhead * 40))
        self.ivf_crossover = int(self.hnsw_crossover * 12)
        self.calibrated = True
        self.detail = {
            "flat_ns_per_point": round(flat_ns, 2),
            "graph_ns_per_hop": round(hop_ns, 2),
            "interpreter_overhead_x": round(overhead, 1),
            "hnsw_crossover": self.hnsw_crossover,
            "ivf_crossover": self.ivf_crossover,
        }
        return self

    def choose(self, count: int, memory_pressure: float = 0.0) -> Strategy:
        """Pick a strategy for a collection of this size under this pressure."""
        if memory_pressure > 0.85 and count > 2_000:
            return Strategy.IVF_PQ                # RAM is the binding constraint
        if count >= self.ivf_crossover:
            return Strategy.IVF_PQ
        if count >= self.hnsw_crossover:
            return Strategy.HNSW
        return Strategy.FLAT

    def as_dict(self) -> dict[str, float]:
        return {"calibrated": self.calibrated, **self.detail}


class AdaptiveVectorIndex:
    """One collection's dense index, able to change shape underneath itself."""

    MIN_TRAIN = 256

    def __init__(self, dim: int, cost: CostModel, name: str = "default") -> None:
        self.dim = dim
        self.cost = cost
        self.name = name
        self.strategy = Strategy.FLAT
        self.ids: list[str] = []
        self.pos: dict[str, int] = {}
        self.matrix = np.zeros((0, dim), dtype=np.float32)
        self.hnsw: HnswIndex | None = None
        self.ivf: IvfPqIndex | None = None
        self.migrations: list[tuple[float, str, int]] = []
        self.searches = 0
        self.exact_searches = 0

    # -- writes -----------------------------------------------------------

    def add(self, point_id: str, vector: np.ndarray) -> None:
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector)) or 1.0
        vector = vector / norm
        if point_id in self.pos:
            self.matrix[self.pos[point_id]] = vector
        else:
            self.pos[point_id] = len(self.ids)
            self.ids.append(point_id)
            self.matrix = (np.vstack([self.matrix, vector]) if len(self.matrix)
                           else vector.reshape(1, -1).copy())
        if self.hnsw is not None:
            self.hnsw.add(point_id, vector)
        if self.ivf is not None and self.ivf.trained:
            self.ivf.add(point_id, vector)
        self._maybe_migrate()

    def remove(self, point_id: str) -> bool:
        index = self.pos.pop(point_id, None)
        if index is None:
            return False
        self.ids.pop(index)
        self.matrix = np.delete(self.matrix, index, axis=0)
        for pid, position in self.pos.items():
            if position > index:
                self.pos[pid] = position - 1
        if self.hnsw is not None:
            self.hnsw.remove(point_id)
        if self.ivf is not None:
            self.ivf.remove(point_id)
        return True

    # -- strategy ---------------------------------------------------------

    def _maybe_migrate(self, memory_pressure: float = 0.0) -> None:
        desired = self.cost.choose(len(self.ids), memory_pressure)
        if desired is self.strategy:
            return
        if desired is Strategy.HNSW:
            self.hnsw = HnswIndex(self.dim, HnswParams())
            for point_id, vector in zip(self.ids, self.matrix):
                self.hnsw.add(point_id, vector)
        elif desired is Strategy.IVF_PQ:
            if len(self.ids) < self.MIN_TRAIN:
                return                                 # not enough to train on yet
            lists = max(4, min(256, int(len(self.ids) ** 0.5)))
            subspaces = max(2, min(16, self.dim // 16))
            while self.dim % subspaces:
                subspaces -= 1
            self.ivf = IvfPqIndex(self.dim, lists=lists, nprobe=max(2, lists // 4),
                                  params=PqParams(subspaces=subspaces))
            self.ivf.train(self.matrix)
            for point_id, vector in zip(self.ids, self.matrix):
                self.ivf.add(point_id, vector)
        self.migrations.append((time.time(), desired.value, len(self.ids)))
        self.strategy = desired

    def force(self, strategy: Strategy) -> None:
        """Override for benchmarking and for the recall harness."""
        self.strategy = Strategy.FLAT
        self.hnsw = self.ivf = None
        if strategy is Strategy.HNSW:
            self.hnsw = HnswIndex(self.dim, HnswParams())
            for point_id, vector in zip(self.ids, self.matrix):
                self.hnsw.add(point_id, vector)
            self.strategy = strategy
        elif strategy is Strategy.IVF_PQ and len(self.ids) >= self.MIN_TRAIN:
            lists = max(4, min(256, int(len(self.ids) ** 0.5)))
            subspaces = max(2, min(16, self.dim // 16))
            while self.dim % subspaces:
                subspaces -= 1
            self.ivf = IvfPqIndex(self.dim, lists=lists, nprobe=max(2, lists // 4),
                                  params=PqParams(subspaces=subspaces))
            self.ivf.train(self.matrix)
            for point_id, vector in zip(self.ids, self.matrix):
                self.ivf.add(point_id, vector)
            self.strategy = strategy

    # -- reads ------------------------------------------------------------

    def search(self, query: np.ndarray, k: int, allow: set[str] | None = None,
               exact: bool = False) -> list[tuple[str, float]]:
        """`allow` is a pre-filter: an id set the planner decided to restrict to."""
        if not self.ids:
            return []
        self.searches += 1
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(query)) or 1.0
        query = query / norm

        if allow is not None:
            rows = [self.pos[p] for p in allow if p in self.pos]
            if not rows:
                return []
            scores = self.matrix[rows] @ query
            order = np.argsort(-scores)[:k]
            self.exact_searches += 1
            return [(self.ids[rows[i]], float(scores[i])) for i in order]

        if exact or self.strategy is Strategy.FLAT:
            self.exact_searches += int(exact)
            scores = self.matrix @ query
            order = np.argsort(-scores)[:k]
            return [(self.ids[i], float(scores[i])) for i in order]

        if self.strategy is Strategy.HNSW and self.hnsw is not None:
            return self.hnsw.search(query, k)
        if self.strategy is Strategy.IVF_PQ and self.ivf is not None:
            return self.ivf.search(query, k)

        scores = self.matrix @ query
        order = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i])) for i in order]

    # -- reporting --------------------------------------------------------

    def __len__(self) -> int:
        return len(self.ids)

    def snapshot(self) -> dict[str, object]:
        out: dict[str, object] = {
            "collection": self.name, "strategy": self.strategy.value, "points": len(self.ids),
            "searches": self.searches, "exact_searches": self.exact_searches,
            "migrations": [{"at": t, "to": s, "points": n} for t, s, n in self.migrations[-5:]],
            "resident_bytes": int(self.matrix.nbytes),
        }
        if self.hnsw is not None:
            out["hnsw"] = self.hnsw.snapshot()
        if self.ivf is not None:
            out["ivf_pq"] = self.ivf.snapshot()
        return out
