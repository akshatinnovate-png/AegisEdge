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
from ..core.tracing import TRACER
from ..memory.filters import Filter
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
    plan: dict[str, Any] = field(default_factory=dict)
    understanding: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"query": self.query, "results": self.results, "stages": self.stages,
                "escalated": self.escalated, "escalation": self.escalation,
                "cached": self.cached, "latency_ms": round(self.latency_ms, 2), "mode": self.mode,
                "plan": self.plan, "understanding": self.understanding, "trace": self.trace}


class RetrievalPipeline:
    def __init__(self, *, store: MemoryStore, sparse, reranker, triton, oracle,
                 bus: EventBus, half_life_days: float = 21.0,
                 understanding=None, late_interaction=None, adapter=None) -> None:
        self.store = store
        self.sparse = sparse
        self.reranker = reranker
        self.triton = triton
        self.oracle = oracle
        self.bus = bus
        self.scorer = Scorer(half_life_days)
        self.cache = SemanticCache()
        self.understanding = understanding
        self.late_interaction = late_interaction
        self.adapter = adapter
        self.queries = 0
        self.rewritten = 0

    async def search(self, query: str, k: int = 5, collection: str = "*",
                     mode: str = "hybrid", explain: bool = True,
                     allow_escalation: bool = True,
                     filters: dict[str, Any] | None = None,
                     understand: bool = True) -> RetrievalResult:
        t_start = time.perf_counter()
        self.queries += 1
        result = RetrievalResult(query=query, mode=mode)
        span_root = TRACER.span("search", query=query, k=k, collection=collection, mode=mode)
        root = span_root.__enter__()

        # 0. understand: repair spelling against the local vocabulary, expand
        #    from corpus co-occurrence, and lift any filter the query implies
        search_text = query
        if understand and self.understanding is not None:
            with TRACER.span("understand") as span:
                analysis = self.understanding.analyse(query)
                result.understanding = analysis.as_dict()
                span.set(corrections=len(analysis.corrections), expansions=len(analysis.expansions))
            if analysis.corrections or analysis.expansions:
                search_text = analysis.normalized
                self.rewritten += 1
            if analysis.filters and not filters:
                filters = analysis.filters
            if collection == "*" and analysis.collection != "*":
                collection = analysis.collection
            result.stages["understand_ms"] = round(analysis.ms, 3)

        # 1. embed the query (micro-batched with concurrent ingest)
        t0 = time.perf_counter()
        with TRACER.span("embed"):
            vector = np.asarray(await self.store.embedder.embed(search_text), dtype=np.float32)
        result.stages["embed_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        # 1b. plan: which of pre-filter / post-filter / scan is cheapest here
        spec = Filter.parse(filters)
        allow: set[str] | None = None
        if hasattr(self.store.store, "plan"):
            t0 = time.perf_counter()
            with TRACER.span("plan") as span:
                plan, allow = self.store.store.plan(collection, spec, k)
                span.set(**plan.as_dict())
            result.plan = plan.as_dict()
            result.stages["plan_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            if plan.kind.value == "empty":
                result.latency_ms = (time.perf_counter() - t_start) * 1000
                result.trace = root.as_dict()
                span_root.__exit__(None, None, None)
                return result

        cached = self.cache.get(vector) if not filters else None
        if cached is not None:
            span_root.__exit__(None, None, None)
            fields = {"results", "escalated", "escalation", "mode", "plan"}
            out = RetrievalResult(query=query, **{k: v for k, v in cached.items() if k in fields})
            out.cached = True
            out.latency_ms = (time.perf_counter() - t_start) * 1000
            # Two differently-spelled queries can normalise to the same text and
            # so share a cache entry. The *answer* is legitimately shared; the
            # explanation of how this query got there is not, so this query's
            # own understanding and trace are carried, never the stored one's.
            out.understanding = result.understanding
            out.trace = root.as_dict()
            out.stages = {**cached.get("stages", {}), **result.stages,
                          "cache_similarity": cached.get("cache_similarity", 1.0)}
            METRICS.incr("retrieval.cache_hits")
            return out

        # 2. recall from both spaces
        t0 = time.perf_counter()
        fetch = max(k * 6, 24)
        fetch = max(fetch, result.plan.get("fetch_k", fetch)) if result.plan else fetch
        with TRACER.span("recall") as recall_span:
            with TRACER.span("dense") as span:
                dense = (self.store.store.search_dense(collection, vector, fetch, allow=allow)
                         if mode != "sparse" else [])
                span.set(hits=len(dense))
            result.stages["dense_ms"] = round((time.perf_counter() - t0) * 1000, 3)

            t0 = time.perf_counter()
            with TRACER.span("sparse") as span:
                sparse_query = self.sparse.encode(search_text)
                sparse = (self.store.store.search_sparse(collection, sparse_query, fetch, allow=allow)
                          if mode != "dense" else [])
                span.set(hits=len(sparse), terms=len(sparse_query))
            result.stages["sparse_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            recall_span.set(fetch=fetch, allow=len(allow) if allow is not None else None)

        # 3. fuse the two orderings
        t0 = time.perf_counter()
        fused = reciprocal_rank_fusion(dense, sparse)[: max(k * 3, 12)]
        result.stages["fusion_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        candidates: list[tuple[str, str, float]] = []
        post_filtering = bool(filters) and allow is None
        for hit in fused:
            point = self.store.points.get(hit.point_id)
            if point is None:
                continue
            if post_filtering and not spec.matches(self.store.store._payload_of(point)):
                continue                                  # post-filter: discard non-matches
            candidates.append((point.id, point.text, hit.rrf * 60))
        if not candidates:
            result.latency_ms = (time.perf_counter() - t_start) * 1000
            return result

        # 4. cross-encoder rerank on-device
        t0 = time.perf_counter()
        with TRACER.span("rerank") as span:
            reranked = self.reranker.rerank(query, candidates, top_k=max(k * 2, 8))
            span.set(candidates=len(candidates))
        result.stages["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        # 4b. late interaction: per-token MaxSim over the shortlist only
        if self.late_interaction is not None and len(reranked) > 1:
            t0 = time.perf_counter()
            with TRACER.span("late_interaction") as span:
                maxsim = self.late_interaction.score(query, [pid for pid, _ in reranked],
                                                     explain=explain)
                span.set(scored=len(maxsim))
            if maxsim:
                blended = {e.point_id: e.score for e in maxsim}
                alignments = {e.point_id: e.as_dict()["alignments"] for e in maxsim}
                reranked = [(pid, 0.6 * score + 0.4 * blended.get(pid, score))
                            for pid, score in reranked]
                reranked.sort(key=lambda row: -row[1])
                result.stages["maxsim_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            else:
                alignments = {}
        else:
            alignments = {}

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

        # 5b. the on-device adapter, trained from this node's own feedback
        adapter_deltas: dict[str, float] = {}
        if self.adapter is not None and len(scored) > 1:
            with TRACER.span("adapter") as span:
                candidates_for_adapter = [
                    (point.id, final, np.asarray(point.dense, dtype=np.float32))
                    for point, final, _, _ in scored if point.dense
                ]
                adapted = self.adapter.rescore(vector, candidates_for_adapter)
                order = {pid: rank for rank, (pid, _, _) in enumerate(adapted)}
                adapter_deltas = {pid: delta for pid, _, delta in adapted}
                scored.sort(key=lambda row: order.get(row[0].id, 1 << 20))
                span.set(reordered=len(adapted))

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
                                    "rrf": round(hit.rrf, 6) if hit else 0.0,
                                    "adapter_delta": round(adapter_deltas.get(point.id, 0.0), 5)}
                if alignments.get(point.id):
                    entry["explain"]["maxsim_alignments"] = alignments[point.id]
            result.results.append(entry)

        result.latency_ms = (time.perf_counter() - t_start) * 1000
        root.set(hits=len(result.results), escalated=result.escalated)
        span_root.__exit__(None, None, None)
        result.trace = root.as_dict()
        METRICS.observe("retrieval.query_ms", result.latency_ms)
        METRICS.incr("retrieval.queries")
        if not filters:
            self.cache.put(vector, result.as_dict())      # filtered results are not cacheable by vector alone
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
        out: dict[str, Any] = {
            "queries": self.queries, "rewritten": self.rewritten,
            "cache": self.cache.snapshot(),
            "latency_ms": hist.snapshot() if hist else {},
        }
        if self.understanding is not None:
            out["understanding"] = self.understanding.snapshot()
        if self.late_interaction is not None:
            out["late_interaction"] = self.late_interaction.snapshot()
        if self.adapter is not None:
            out["adapter"] = self.adapter.snapshot()
        return out
