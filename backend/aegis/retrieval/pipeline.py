"""Hybrid retrieval pipeline.

embed → dense + sparse recall → RRF → rerank → final scoring → optional
Triton escalation. Every stage's latency and every score's components are
recorded, so a result can be explained rather than merely returned.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.bus import EventBus
from ..core.metrics import METRICS
from ..memory.schema import MemoryPoint, Sensitivity
from ..memory.store import MemoryStore
from .cache import SemanticCache
from .fusion import reciprocal_rank_fusion
from .scoring import Scorer

COMPLEX_MARKERS = ("why", "compare", "explain", "summarise", "summarize", "trend", "root cause", "how many")


@dataclass(slots=True)
class RetrievalResult:
    query: str
    results: list[dict[str, Any]] = field(default_factory=list)
    stages: dict[str, float] = field(default_factory=dict)
    escalated: bool = False
    escalation: dict[str, Any] = field(default_factory=dict)
    cached: bool = False
    latency_ms: float = 0.0
    mode: str = "hybrid"

    def as_dict(self) -> dict[str, Any]:
        return {"query": self.query, "results": self.results, "stages": self.stages,
                "escalated": self.escalated, "escalation": self.escalation,
                "cached": self.cached, "latency_ms": round(self.latency_ms, 2), "mode": self.mode}


class RetrievalPipeline:
    def __init__(self, *, store: MemoryStore, sparse, reranker, triton, oracle,
                 bus: EventBus, half_life_days: float = 21.0) -> None:
        self.store = store
        self.sparse = sparse
        self.reranker = reranker
        self.triton = triton
        self.oracle = oracle
        self.bus = bus
        self.scorer = Scorer(half_life_days)
        self.cache = SemanticCache()
        self.queries = 0

    async def search(self, query: str, k: int = 5, collection: str = "*",
                     mode: str = "hybrid", explain: bool = True,
                     allow_escalation: bool = True) -> RetrievalResult:
        t_start = time.perf_counter()
        self.queries += 1
        result = RetrievalResult(query=query, mode=mode)

        # 1. embed the query (micro-batched with concurrent ingest)
        t0 = time.perf_counter()
        vector = np.asarray(await self.store.embedder.embed(query), dtype=np.float32)
        result.stages["embed_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        cached = self.cache.get(vector)
        if cached is not None:
            fields = {"results", "stages", "escalated", "escalation", "mode"}
            out = RetrievalResult(query=query, **{k: v for k, v in cached.items() if k in fields})
            out.cached = True
            out.latency_ms = (time.perf_counter() - t_start) * 1000
            out.stages = {**out.stages, "cache_similarity": cached.get("cache_similarity", 1.0)}
            METRICS.incr("retrieval.cache_hits")
            return out

        # 2. recall from both spaces
        t0 = time.perf_counter()
        fetch = max(k * 6, 24)
        dense = self.store.store.search_dense(collection, vector, fetch) if mode != "sparse" else []
        result.stages["dense_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        t0 = time.perf_counter()
        sparse_query = self.sparse.encode(query)
        sparse = self.store.store.search_sparse(collection, sparse_query, fetch) if mode != "dense" else []
        result.stages["sparse_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        # 3. fuse the two orderings
        t0 = time.perf_counter()
        fused = reciprocal_rank_fusion(dense, sparse)[: max(k * 3, 12)]
        result.stages["fusion_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        candidates: list[tuple[str, str, float]] = []
        for hit in fused:
            point = self.store.points.get(hit.point_id)
            if point is None:
                continue
            candidates.append((point.id, point.text, hit.rrf * 60))
        if not candidates:
            result.latency_ms = (time.perf_counter() - t_start) * 1000
            return result

        # 4. cross-encoder rerank on-device
        t0 = time.perf_counter()
        reranked = self.reranker.rerank(query, candidates, top_k=max(k * 2, 8))
        result.stages["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        # 5. final scoring with recency / salience / confidence
        scored: list[tuple[MemoryPoint, float, dict[str, float], float]] = []
        rrf_by_id = {h.point_id: h for h in fused}
        for point_id, rerank_score in reranked:
            point = self.store.points.get(point_id)
            if point is None:
                continue
            final, breakdown = self.scorer.score(point, rerank_score)
            scored.append((point, final, breakdown, rerank_score))
        scored.sort(key=lambda x: -x[1])
        top = scored[:k]

        # 6. decide whether the cloud tier has anything to add
        top_score = top[0][1] if top else 0.0
        may_egress = all(p.sensitivity is not Sensitivity.RESTRICTED for p, *_ in top)
        decision = self.triton.should_escalate(
            top_score=top_score,
            link_state=self.oracle.state.value,
            rtt_ms=self.oracle.rtt_ms,
            may_egress=may_egress and allow_escalation,
            complex_query=any(m in query.lower() for m in COMPLEX_MARKERS),
        )
        result.escalation = decision.as_dict()
        if decision.escalate:
            try:
                t0 = time.perf_counter()
                remote = await self.triton.rerank(query, [(p.id, p.text) for p, *_ in top])
                result.stages["triton_ms"] = round((time.perf_counter() - t0) * 1000, 3)
                order = {pid: score for pid, score in remote}
                top.sort(key=lambda row: -order.get(row[0].id, 0.0))
                result.escalated = True
            except Exception as exc:
                result.escalation["error"] = str(exc)   # local answer already stands

        for point, final, breakdown, rerank_score in top:
            point.touch()
            hit = rrf_by_id.get(point.id)
            entry = {
                "id": point.id, "collection": point.collection, "text": point.text,
                "score": round(final, 6), "tier": point.tier.value,
                "sensitivity": point.sensitivity.value, "confidence": round(point.confidence, 3),
                "age_s": round(point.age_s(), 1), "stale": point.stale,
                "superseded_by": point.superseded_by,
                "matched_by": hit.contributors if hit else [],
            }
            if explain:
                entry["explain"] = {"rerank": round(rerank_score, 4), **breakdown,
                                    "rrf": round(hit.rrf, 6) if hit else 0.0}
            result.results.append(entry)

        result.latency_ms = (time.perf_counter() - t_start) * 1000
        METRICS.observe("retrieval.query_ms", result.latency_ms)
        METRICS.incr("retrieval.queries")
        self.cache.put(vector, result.as_dict())
        self.bus.publish(
            "search", "query", query=query, hits=len(result.results),
            latency_ms=round(result.latency_ms, 2), escalated=result.escalated,
            message=(f"\"{query}\" → <b>{len(result.results)}</b> hits in "
                     f"{result.latency_ms:.1f} ms"
                     + (" · <b>escalated</b>" if result.escalated else " · local")),
        )
        return result

    def snapshot(self) -> dict[str, Any]:
        hist = METRICS.histograms.get("retrieval.query_ms")
        return {"queries": self.queries, "cache": self.cache.snapshot(),
                "latency_ms": hist.snapshot() if hist else {}}
