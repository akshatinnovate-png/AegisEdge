"""CollectionIndex — one collection's complete retrieval surface.

Composes, rather than reimplements: full-precision vectors live once in
`VectorStorage` (resident, or memmapped for the cold tail), the ANN strategy
is chosen by the calibrated cost model, the cold tier is scanned as 1-bit
codes and rescored by paging in only the shortlist, sparse postings run
alongside for rare-token recall, and payload statistics feed the planner.

Tiers here are *encodings*, not copies:

    HOT   full precision, resident, in the ANN index
    WARM  int8 scalar codes, resident, rescored from the same vectors
    COLD  1-bit codes resident, vectors on disk, rescored by page-in
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .ann import AdaptiveVectorIndex, CostModel, Strategy
from .filters import Filter, PayloadIndex
from .planner import PlanKind, QueryPlan, QueryPlanner
from .quantize import ScalarQuantizer
from .rabitq import ColdCodebook, RaBitQ, adaptive_shortlist
from .schema import Tier
from .vectors import VectorStorage


class SparseIndex:
    """Inverted index over hashed term ids with impact-ordered postings."""

    def __init__(self) -> None:
        self.postings: dict[int, dict[str, float]] = defaultdict(dict)
        self.norms: dict[str, float] = {}
        self.terms_of: dict[str, list[int]] = {}

    def add(self, point_id: str, sparse: dict[int, float]) -> None:
        self.remove(point_id)
        norm = 0.0
        for term, weight in sparse.items():
            self.postings[term][point_id] = weight
            norm += weight * weight
        self.terms_of[point_id] = list(sparse)
        self.norms[point_id] = norm ** 0.5 or 1.0

    def remove(self, point_id: str) -> None:
        for term in self.terms_of.pop(point_id, ()):        # O(terms), not O(vocabulary)
            bucket = self.postings.get(term)
            if bucket is not None:
                bucket.pop(point_id, None)
                if not bucket:
                    del self.postings[term]
        self.norms.pop(point_id, None)

    def search(self, sparse: dict[int, float], k: int, allow: set[str] | None = None) -> list[tuple[str, float]]:
        if not sparse:
            return []
        accumulator: dict[str, float] = defaultdict(float)
        query_norm = (sum(w * w for w in sparse.values()) ** 0.5) or 1.0
        for term, query_weight in sparse.items():
            for point_id, weight in self.postings.get(term, {}).items():
                if allow is not None and point_id not in allow:
                    continue
                accumulator[point_id] += query_weight * weight
        scored = [(pid, s / (query_norm * self.norms.get(pid, 1.0))) for pid, s in accumulator.items()]
        scored.sort(key=lambda x: -x[1])
        return scored[:k]

    def snapshot(self) -> dict[str, int]:
        return {"terms": len(self.postings), "documents": len(self.norms),
                "postings": sum(len(b) for b in self.postings.values())}


INDEXED_FIELDS = ("collection", "sensitivity", "sync_class", "model_version",
                  "device_id", "ts", "confidence", "stale", "pinned", "source")


class CollectionIndex:
    def __init__(self, dim: int, cost: CostModel, name: str = "default",
                 data_dir: Path | None = None) -> None:
        self.dim = dim
        self.name = name
        cold_path = (Path(data_dir) / f"{name}.cold.f32") if data_dir else None
        self.storage = VectorStorage(dim, cold_path)
        self.ann = AdaptiveVectorIndex(dim, cost, name)
        self.sparse = SparseIndex()
        self.payload = PayloadIndex(INDEXED_FIELDS)
        self.planner = QueryPlanner(self.payload)

        self.tier_of: dict[str, Tier] = {}
        self.warm_codes: dict[str, tuple[np.ndarray, float]] = {}
        # Cold codes are RaBitQ: packed bits plus the two scalars that make
        # them an estimate rather than a proxy. Eight bytes per vector buys
        # an error bound, and the bound is what sets the rescoring depth.
        self.rabitq = RaBitQ(dim)
        self.cold_codes = ColdCodebook(self.rabitq)
        self.rescored = 0
        self.cold_scans = 0
        self.bounded_shortlists = 0
        self.budget_capped = 0
        self.last_plan: QueryPlan | None = None

    # -- writes -----------------------------------------------------------

    def upsert(self, point_id: str, dense: np.ndarray, sparse: dict[int, float],
               tier: Tier, payload: dict[str, Any] | None = None) -> None:
        vector = np.asarray(dense, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector)) or 1.0
        vector = vector / norm

        self.storage.put(point_id, vector)
        self.sparse.add(point_id, sparse)
        if payload is not None:
            self.payload.index(point_id, payload)
        self.tier_of[point_id] = tier
        self._encode_for(point_id, vector, tier)
        if tier is Tier.COLD:
            self.ann.remove(point_id)
            self.storage.evict(point_id)
        else:
            self.ann.add(point_id, vector)

    def _encode_for(self, point_id: str, vector: np.ndarray, tier: Tier) -> None:
        self.warm_codes.pop(point_id, None)
        self.cold_codes.pop(point_id)
        if tier is Tier.WARM:
            codes, scales = ScalarQuantizer.encode(vector.reshape(1, -1))
            self.warm_codes[point_id] = (codes[0], float(scales[0][0]))
        elif tier is Tier.COLD:
            # The centroid tracks the corpus; a vector encoded against a stale
            # one is still decodable, only less sharp, which `centroid_drift`
            # reports so the maintenance lane can re-encode when it matters.
            self.cold_codes.put(point_id, vector)

    def move(self, point_id: str, tier: Tier) -> bool:
        current = self.tier_of.get(point_id)
        if current is None or current is tier:
            return False
        if current is Tier.COLD:
            self.storage.promote(point_id)
        vector = self.storage.get(point_id)
        if vector is None:
            return False
        vector = np.asarray(vector, dtype=np.float32)
        self.tier_of[point_id] = tier
        self._encode_for(point_id, vector, tier)
        if tier is Tier.COLD:
            self.ann.remove(point_id)
            self.storage.evict(point_id)
        else:
            self.ann.add(point_id, vector)
        return True

    def remove(self, point_id: str) -> None:
        self.tier_of.pop(point_id, None)
        self.warm_codes.pop(point_id, None)
        self.cold_codes.pop(point_id)
        self.ann.remove(point_id)
        self.sparse.remove(point_id)
        self.payload.drop(point_id)
        self.storage.drop(point_id)

    # -- reads ------------------------------------------------------------

    def migrate_pending(self) -> dict[str, Any] | None:
        """Run any index rebuild the write path deferred."""
        return self.ann.migrate_pending()

    def plan(self, spec: Filter, k: int) -> QueryPlan:
        plan = self.planner.plan(spec, len(self), k,
                                 ann_available=self.ann.strategy is not Strategy.FLAT)
        self.last_plan = plan
        return plan

    def search_dense(self, query: np.ndarray, k: int, allow: set[str] | None = None,
                     exact: bool = False) -> list[tuple[str, float]]:
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(query)) or 1.0
        query = query / norm

        results = self.ann.search(query, k, allow=allow, exact=exact)
        cold = self._search_cold(query, k, allow)
        if not cold:
            return results[:k]
        merged = results + cold
        merged.sort(key=lambda x: -x[1])
        seen: set[str] = set()
        out: list[tuple[str, float]] = []
        for point_id, score in merged:
            if point_id in seen:
                continue
            seen.add(point_id)
            out.append((point_id, score))
            if len(out) >= k:
                break
        return out

    def _search_cold(self, query: np.ndarray, k: int, allow: set[str] | None,
                     budget: int | None = None) -> list[tuple[str, float]]:
        """Scan 1-bit codes in RAM, then page in only what the bound cannot rule out.

        The old shape of this was a fixed 6x over-fetch: scan, take the best
        6k by an unbounded proxy score, rescore those. Six was a guess, and a
        guess is the wrong thing to put between a query and its answer.

        RaBitQ gives every candidate an unbiased estimate and a two-sided
        error bound, so the shortlist is derived instead: the k-th largest
        lower bound is a score the true top-k provably reaches, and anything
        whose upper bound falls short of it cannot belong there and is never
        fetched from disk. Easy queries collapse to a handful of reads;
        ambiguous ones widen on their own, up to the budget the caller allows.
        """
        if len(self.cold_codes) == 0:
            return []
        if allow is None:
            ids, codes = self.cold_codes.ids(), self.cold_codes.codes()
        else:
            ids, codes = self.cold_codes.subset(allow)
        if not ids:
            return []
        self.cold_scans += 1
        estimate, bound = self.rabitq.estimate(self.rabitq.prepare(query), codes)

        ceiling = budget if budget is not None else max(k * 32, 256)
        shortlist = adaptive_shortlist(estimate, bound, k, budget=ceiling)
        self.bounded_shortlists += 1
        if shortlist.size >= ceiling:
            # The bound stopped being informative and the latency budget took
            # over. Recorded, because a guarantee that silently lapsed is
            # worse than never having claimed one.
            self.budget_capped += 1

        wanted = [ids[i] for i in shortlist]
        vectors = self.storage.gather(wanted)               # the only disk touch
        exact = vectors @ query
        self.rescored += len(wanted)
        order = np.argsort(-exact)[:k]
        return [(wanted[i], float(exact[i])) for i in order]

    def search_sparse(self, sparse: dict[int, float], k: int,
                      allow: set[str] | None = None) -> list[tuple[str, float]]:
        return self.sparse.search(sparse, k, allow)

    # -- reporting --------------------------------------------------------

    def counts(self) -> dict[str, int]:
        out = {t.value: 0 for t in (Tier.HOT, Tier.WARM, Tier.COLD)}
        for tier in self.tier_of.values():
            if tier.value in out:
                out[tier.value] += 1
        return out

    def ids(self) -> Iterable[str]:
        return self.tier_of.keys()

    def __len__(self) -> int:
        return len(self.tier_of)

    def snapshot(self) -> dict[str, Any]:
        counts = self.counts()
        code_bytes = ((len(self.warm_codes) * (self.dim + 4))
                      + self.cold_codes.nbytes)
        return {
            "collection": self.name, "points": len(self), "tiers": counts,
            "ann": self.ann.snapshot(), "sparse": self.sparse.snapshot(),
            "storage": self.storage.snapshot(), "payload": self.payload.snapshot(),
            "planner": {"plans": self.planner.plans, "by_kind": dict(self.planner.by_kind)},
            "code_bytes": code_bytes, "rescored": self.rescored, "cold_scans": self.cold_scans,
            "cold_codec": self.rabitq.snapshot(),
            "bounded_shortlists": self.bounded_shortlists,
            "budget_capped": self.budget_capped,
            "last_plan": self.last_plan.as_dict() if self.last_plan else None,
        }


# Backwards-compatible alias: the tiered index is now a full collection index.
TieredIndex = CollectionIndex
