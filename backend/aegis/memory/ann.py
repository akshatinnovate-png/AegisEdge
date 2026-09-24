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

import heapq
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from .growable import GrowableMatrix
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
        """Measure both search shapes on *this* device, then pick the crossover.

        The previous version of this measured a graph hop as
        ``data[neighbours] @ query`` — thirty-two neighbours in a single
        vectorised call. That is the arithmetic of a hop and almost none of
        its cost. A real traversal in the interpreter pays per node for the
        visited set, the candidate heap and the Python-level loop around them,
        and those dominate. Measuring only the arithmetic put the overhead
        ratio at about 1 and collapsed the crossover onto its 5,000 floor.

        `scripts/strategy_bakeoff.py` shows what that cost, by forcing each
        strategy onto the same corpus:

            20,000 points   flat    p50 0.886 ms   recall 1.000   build   0 s
                            hnsw    p50 4.340 ms   recall 0.773   build 252 s
                            ivf_pq  p50 550.5 ms   recall 0.997   build 123 s

        Exhaustive search was 4.9x faster than the graph with perfect recall
        and no build at all, and the model was choosing the graph from 5,000
        points up. It only failed to make anything worse because index
        migration is deferred to the maintenance lane, which the scale runs
        never reach — a latent misconfiguration masked by an unrelated fix.

        So the hop is now timed with its bookkeeping, and the crossover is
        solved from the two measurements rather than a constant.
        """
        rng = np.random.default_rng(0)
        data = rng.normal(size=(sample, dim)).astype(np.float32)
        data /= np.linalg.norm(data, axis=1, keepdims=True)
        query = data[0]

        t0 = time.perf_counter()
        for _ in range(5):
            np.argsort(-(data @ query))[:10]
        flat_ns = (time.perf_counter() - t0) / (5 * sample) * 1e9

        # A hop as it is actually executed: dedupe against the visited set,
        # gather, score, and push onto the candidate heap.
        fan_out = 32
        rounds = 200
        adjacency = [rng.integers(0, sample, size=fan_out).tolist() for _ in range(rounds)]
        visited: set[int] = set()
        heap: list[tuple[float, int]] = []
        t0 = time.perf_counter()
        for neighbours in adjacency:
            fresh = [n for n in neighbours if n not in visited]
            if not fresh:
                continue
            visited.update(fresh)
            scores = data[fresh] @ query
            for node, score in zip(fresh, scores):
                heapq.heappush(heap, (-float(score), node))
            if len(visited) > sample // 2:
                visited.clear()
                heap.clear()
        hop_ns = (time.perf_counter() - t0) / (rounds * fan_out) * 1e9

        self.flat_ns_per_point = flat_ns
        self.graph_ns_per_hop = hop_ns
        overhead = max(hop_ns / max(flat_ns, 1e-9), 1.0)

        # Flat costs n * flat_ns. A graph walk touches roughly ef * log2(n)
        # nodes, each at hop_ns. The crossover is the smallest n where the walk
        # is genuinely cheaper — and it has to beat exhaustive search that is
        # also exact, so a graph only earns the switch once it is clearly ahead.
        self.hnsw_crossover = self._solve_crossover(flat_ns, hop_ns)
        # IVF-PQ is not a latency structure here: the bake-off puts it at
        # 550 ms against flat's 0.9 ms at 20,000 points. It exists for the case
        # where RAM, not time, is the binding constraint, which `choose` treats
        # as a separate decision. The count-based threshold is kept far out of
        # the way so it never wins on size alone.
        self.ivf_crossover = int(self.hnsw_crossover * 12)
        self.calibrated = True
        self.detail = {
            "flat_ns_per_point": round(flat_ns, 2),
            "graph_ns_per_hop": round(hop_ns, 2),
            "interpreter_overhead_x": round(overhead, 1),
            "hnsw_crossover": self.hnsw_crossover,
            "ivf_crossover": self.ivf_crossover,
            "validated_by": "scripts/strategy_bakeoff.py",
        }
        return self

    # How many nodes a search actually scores, as a multiple of ef*log2(n).
    # Derived, not guessed: the bake-off measured HNSW at 4.34 ms per query on
    # 20,000 points where a hop costs 852.8 ns, so the walk scored about 5,090
    # nodes against ef*log2(20,000) = 915 — a factor of 5.6. Assuming the
    # textbook ef*log2(n) directly overestimates the walk by that much and
    # pushes the crossover out to 880,000 points, which is its own kind of
    # wrong. Re-derive it with `scripts/strategy_bakeoff.py` on new hardware.
    VISIT_FACTOR = 5.6

    @classmethod
    def _solve_crossover(cls, flat_ns: float, hop_ns: float, ef: int = 64,
                         margin: float = 1.5) -> int:
        """Smallest corpus where a graph walk beats the linear scan by `margin`.

        The margin is not timidity. Exhaustive search returns recall 1.000; the
        graph measured 0.773 at 20,000 points. Trading exactness for latency is
        only worth it when the latency win is decisive, so a dead heat resolves
        in favour of the exact answer.
        """
        for n in range(10_000, 20_000_001, 10_000):
            flat_cost = n * flat_ns
            walk_cost = ef * math.log2(max(n, 2)) * cls.VISIT_FACTOR * hop_ns
            if walk_cost * margin < flat_cost:
                return n
        return 20_000_000

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
        self.rows = GrowableMatrix(dim)
        self.hnsw: HnswIndex | None = None
        self.ivf: IvfPqIndex | None = None
        self.migrations: list[tuple[float, str, int]] = []
        self.searches = 0
        self.exact_searches = 0
        # Migration is *decided* on the write path and *performed* off it.
        self.pending_strategy: Strategy | None = None
        self.deferred_migrations = 0
        self.last_migration_s = 0.0

    # -- writes -----------------------------------------------------------

    def add(self, point_id: str, vector: np.ndarray) -> None:
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector)) or 1.0
        vector = vector / norm
        if point_id in self.pos:
            self.rows[self.pos[point_id]] = vector
        else:
            self.pos[point_id] = self.rows.append(vector)
            self.ids.append(point_id)
        if self.hnsw is not None:
            self.hnsw.add(point_id, vector)
        if self.ivf is not None and self.ivf.trained:
            self.ivf.add(point_id, vector)
        self._maybe_migrate()

    def remove(self, point_id: str) -> bool:
        index = self.pos.pop(point_id, None)
        if index is None:
            return False
        # Swap-with-last keeps removal O(1); only the row that moved needs its
        # bookkeeping repointed, rather than every row after the hole.
        moved = self.rows.swap_remove(index)
        if moved is not None:
            moved_id = self.ids[moved]
            self.ids[index] = moved_id
            self.pos[moved_id] = index
        self.ids.pop()
        if self.hnsw is not None:
            self.hnsw.remove(point_id)
        if self.ivf is not None:
            self.ivf.remove(point_id)
        return True

    # -- strategy ---------------------------------------------------------

    @property
    def matrix(self) -> np.ndarray:
        """The populated rows as a contiguous view — still one BLAS call."""
        return self.rows.view

    def _maybe_migrate(self, memory_pressure: float = 0.0) -> None:
        """Decide, on the write path. Do not build, on the write path.

        Building an HNSW graph for a corpus that has just crossed the crossover
        takes minutes in pure Python, and doing it inline stalls every write
        behind it — a node that quietly stops accepting data for four minutes
        because it got popular is worse than one that stays on the slower
        strategy a little longer. The decision is recorded here and the rebuild
        runs in the maintenance lane, where the scheduler can shed it.
        """
        desired = self.cost.choose(len(self.ids), memory_pressure)
        if desired is self.strategy or desired is self.pending_strategy:
            return
        self.pending_strategy = desired
        self.deferred_migrations += 1

    def migrate_pending(self) -> dict[str, Any] | None:
        """Perform a deferred migration. Called from the maintenance lane."""
        desired = self.pending_strategy
        if desired is None or desired is self.strategy:
            self.pending_strategy = None
            return None
        started = time.perf_counter()
        points = len(self.ids)
        self._build(desired)
        elapsed = time.perf_counter() - started
        self.pending_strategy = None
        self.last_migration_s = elapsed
        return {"to": self.strategy.value, "points": points,
                "seconds": round(elapsed, 3)}

    def _build(self, desired: Strategy) -> None:
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
        """Build a strategy synchronously — benchmarks and the recall harness."""
        self.pending_strategy = None
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
            "pending_strategy": self.pending_strategy.value if self.pending_strategy else None,
            "deferred_migrations": self.deferred_migrations,
            "last_migration_s": round(self.last_migration_s, 3),
            "migrations": [{"at": t, "to": s, "points": n} for t, s, n in self.migrations[-5:]],
            "resident_bytes": self.rows.nbytes,
            "buffer": self.rows.snapshot(),
        }
        if self.hnsw is not None:
            out["hnsw"] = self.hnsw.snapshot()
        if self.ivf is not None:
            out["ivf_pq"] = self.ivf.snapshot()
        return out
