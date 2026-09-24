"""Hybrid retrieval pipeline.

embed → dense + sparse recall → RRF → rerank → final scoring → optional
Triton escalation. Every stage's latency and every score's components are
recorded, so a result can be explained rather than merely returned.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.bus import EventBus
from ..core.metrics import METRICS
from ..core.slo import SLOManager
from ..core.tracing import TRACER
from ..memory.filters import Filter


def _filter_key(filters: dict[str, Any] | None) -> str:
    """A stable digest of a filter spec, for use in a cache key.

    Canonical because two spellings of the same restriction must collide: a
    key that depended on dict ordering would miss hits that are genuinely the
    same question, and one that ignored the filter would serve the wrong
    answer confidently.
    """
    if not filters:
        return "-"
    try:
        canonical = json.dumps(filters, sort_keys=True, default=str, separators=(",", ":"))
    except Exception:
        canonical = repr(sorted(filters.items(), key=lambda kv: str(kv[0])))
    return hashlib.blake2s(canonical.encode(), digest_size=8).hexdigest()


def _cpu_seconds() -> float:
    """Process CPU time, which is what a joule estimate has to be built on.

    Wall time bills a query for every millisecond it spent queued behind
    something else; CPU time bills it for the work it actually caused.
    """
    usage = os.times()
    return usage.user + usage.system
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
    confidence: dict[str, Any] = field(default_factory=dict)
    diversity: dict[str, Any] = field(default_factory=dict)
    graph_context: dict[str, Any] = field(default_factory=dict)
    degradation: dict[str, Any] = field(default_factory=dict)
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
                "plan": self.plan, "understanding": self.understanding, "trace": self.trace,
                "confidence": self.confidence, "diversity": self.diversity,
                "graph_context": self.graph_context, "degradation": self.degradation}


class RetrievalPipeline:
    def __init__(self, *, store: MemoryStore, sparse, reranker, triton, oracle,
                 bus: EventBus, half_life_days: float = 21.0,
                 understanding=None, adapter=None,
                 graph=None, conformal=None, diversity=None, slo=None,
                 energy=None) -> None:
        self.store = store
        self.sparse = sparse
        self.reranker = reranker
        self.triton = triton
        self.oracle = oracle
        self.bus = bus
        self.scorer = Scorer(half_life_days)
        self.cache = SemanticCache()
        self.understanding = understanding
        self.adapter = adapter
        self.graph = graph
        self.conformal = conformal
        self.diversity = diversity
        self.slo: SLOManager | None = slo
        self.energy = energy
        self.queries = 0
        self.rewritten = 0
        self.degraded_queries = 0
        self.relaxed_inferences = 0
        self.queries_from_cache = 0

    async def _relax(self, result, inferred, query, k, collection, mode, explain,
                     allow_escalation, filters, tenant_id, stages):
        """Retry at the scope the caller asked for, dropping what we inferred.

        An inference must never be able to empty a result set the caller would
        otherwise have got. Query understanding reads "conveyor vibration
        night shift" as sensor intent and narrows to the sensor collection; if
        the corpus keeps those memories as episodic, that hard filter returns
        nothing, while the single word "conveyor" returns five hits because it
        inferred nothing at all. A guess that silently replaces results with an
        empty page is worse than no guess.

        Only what *this layer* added is ever backed out. A filter the caller
        supplied is honoured even when it matches nothing — that is their
        question, and answering a different one would be the same sin in the
        opposite direction.
        """
        self.relaxed_inferences += 1
        relaxed = await self.search(
            query, k=k, collection=collection, mode=mode, explain=explain,
            allow_escalation=allow_escalation, filters=filters,
            understand=False, tenant_id=tenant_id, _relaxable=False)
        relaxed.understanding = {
            **result.understanding,
            "inference_relaxed": {
                "dropped": inferred,
                "reason": "the inferred narrowing matched nothing; retried at "
                          "the scope the caller asked for",
                "recovered_hits": len(relaxed.results),
            },
        }
        relaxed.stages = {**stages, **relaxed.stages}
        return relaxed

    async def search(self, query: str, k: int = 5, collection: str = "*",
                     mode: str = "hybrid", explain: bool = True,
                     allow_escalation: bool = True,
                     filters: dict[str, Any] | None = None,
                     understand: bool = True,
                     tenant_id: str | None = None,
                     _relaxable: bool = True) -> RetrievalResult:
        t_start = time.perf_counter()
        cpu_start = _cpu_seconds()
        cpu_used = lambda: _cpu_seconds() - cpu_start          # noqa: E731
        # What the caller actually asked for, kept so an inference that turns
        # out to be wrong can be backed out rather than silently obeyed.
        asked_collection, asked_filters = collection, filters
        inferred: dict[str, Any] = {}
        self.queries += 1
        result = RetrievalResult(query=query, mode=mode)
        span_root = TRACER.span("search", query=query, k=k, collection=collection, mode=mode)
        root = span_root.__enter__()

        # Which optional stages may run at all right now. Asking once keeps the
        # answer consistent across the query rather than shifting mid-flight.
        allowed = (lambda feature: self.slo.allows(feature)) if self.slo else (lambda _f: True)
        if self.slo is not None and int(self.slo.level) > 0:
            self.degraded_queries += 1
            result.degradation = {"level": self.slo.level.name,
                                  "disabled": self.slo.disabled()}

        # 0. understand: repair spelling against the local vocabulary, expand
        #    from corpus co-occurrence, and lift any filter the query implies
        search_text = query
        if understand and self.understanding is not None and allowed("query_understanding"):
            with TRACER.span("understand") as span:
                analysis = self.understanding.analyse(query)
                result.understanding = analysis.as_dict()
                span.set(corrections=len(analysis.corrections), expansions=len(analysis.expansions))
            if analysis.corrections or analysis.expansions:
                search_text = analysis.normalized
                self.rewritten += 1
            if analysis.filters and not filters:
                filters = analysis.filters
                inferred["filters"] = analysis.filters
            if collection == "*" and analysis.collection != "*":
                collection = analysis.collection
                inferred["collection"] = analysis.collection
            result.stages["understand_ms"] = round(analysis.ms, 3)

        # 0b. the exact layer, before the encoder. Understanding has already
        #     settled the collection and any inferred filter, so the namespace
        #     is final — and an identical question can now be answered without
        #     running the model at all, which is the whole point of putting it
        #     here rather than after the embed.
        namespace = (f"{tenant_id or 'default'}|{collection}|{mode}|{k}"
                     f"|{_filter_key(filters)}")
        verbatim = self.cache.get_exact(search_text, namespace)
        if verbatim is not None:
            span_root.__exit__(None, None, None)
            fields = {"results", "escalated", "escalation", "mode", "plan"}
            out = RetrievalResult(query=query,
                                  **{key: value for key, value in verbatim.items()
                                     if key in fields})
            out.cached = True
            out.understanding = result.understanding
            out.latency_ms = (time.perf_counter() - t_start) * 1000
            out.stages = {**result.stages, "cache": "exact"}
            self.queries_from_cache += 1
            if self.slo is not None:
                self.slo.observe(out.latency_ms, ok=True)
            return out

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
                # The planner proves the filter matches nothing. When that
                # filter was our own guess, that is the strongest possible
                # signal the guess was wrong — and the earliest point we can
                # know it, which is why the relaxation has to live here too
                # and not only at the end of the happy path.
                if inferred and _relaxable:
                    return await self._relax(
                        result, inferred, query, k, asked_collection, mode, explain,
                        allow_escalation, asked_filters, tenant_id, result.stages)
                return result

        # Everything that changes what a correct answer *is* belongs in the
        # cache key, not just who asked. Keying on the tenant alone meant a
        # query run once over all collections was then served verbatim for
        # `collection="procedural"` — five episodic hits from a collection
        # holding nothing at all — and the same for a different retrieval mode
        # or a larger k. The tenant was namespaced after a cross-tenant leak;
        # the scope of the question was not, and scope is part of the question.
        # Filters belong *in the key*, not in a condition that skips the cache.
        # Guarding on `if not filters` looked conservative and was close to
        # fatal: query understanding infers a collection filter on most
        # queries, so almost nothing was ever cached. Measured over 128
        # queries with 64 repeats: 0 entries, 0 hits, 0 misses — the cache had
        # never been asked a question, let alone answered one.
        #
        # Caching is unsafe only when the key fails to capture what changes
        # the answer. So everything that does goes in: the tenant, the
        # collection, the retrieval mode, k, and a canonical digest of the
        # filter spec.
        cached = self.cache.get(vector, namespace)
        if cached is not None:
            # The vector layer has just decided this text maps to this answer.
            # Teach the exact layer, so the *next* identical question skips the
            # encoder entirely instead of re-deriving the same conclusion. A
            # cache that only learns from full work never learns from itself.
            self.cache.put_exact(search_text, cached, namespace)
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
            out.degradation = result.degradation      # this query's level, not the cached one's
            out.trace = root.as_dict()
            out.stages = {**cached.get("stages", {}), **result.stages,
                          "cache_similarity": cached.get("cache_similarity", 1.0)}
            METRICS.incr("retrieval.cache_hits")
            return out

        # 2. recall from both spaces
        # tenancy: the visible id set is intersected with the plan's allow-set,
        # so a tenant cannot see another tenant's memories through any path
        if tenant_id is not None:
            visible = self.store.visible(tenant_id)
            allow = visible if allow is None else (allow & visible)
            if not allow:
                # Deliberately *not* relaxed. An empty visible set is a tenancy
                # boundary, not an inference that went wrong, and retrying at a
                # wider scope here would turn a correct empty answer into an
                # isolation breach.
                result.latency_ms = (time.perf_counter() - t_start) * 1000
                result.trace = root.as_dict()
                span_root.__exit__(None, None, None)
                return result

        t0 = time.perf_counter()
        fetch = max(k * 6, 24) if allowed("wide_fetch") else max(k * 2, 10)
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
            # Retrieval found nothing to rank. If the scope it searched was our
            # own inference rather than the caller's, this is where that shows.
            if inferred and _relaxable:
                return await self._relax(
                    result, inferred, query, k, asked_collection, mode, explain,
                    allow_escalation, asked_filters, tenant_id, result.stages)
            return result

        # 4. cross-encoder rerank on-device
        t0 = time.perf_counter()
        with TRACER.span("rerank") as span:
            if allowed("cross_encoder"):
                reranked = self.reranker.rerank(query, candidates, top_k=max(k * 2, 8),
                                                explain=explain and allowed("late_interaction"))
            else:
                # Shed the cross-encoder but keep the fusion ordering rather
                # than returning an arbitrary one.
                reranked = sorted(((pid, score) for pid, _, score in candidates),
                                  key=lambda row: -row[1])[: max(k * 2, 8)]
            span.set(candidates=len(candidates), late_interaction=allowed("cross_encoder"))
        result.stages["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        alignments = {
            pid: [a.as_dict() for a in row.alignments]
            for pid, row in getattr(self.reranker, "last_explanations", {}).items()
        } if explain else {}

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

        # 5a. graph boost: memories connected to the query's entities, even
        #     when their text does not look like the query. This is the half
        #     of recall embeddings structurally cannot reach.
        if self.graph is not None and allowed("graph_boost") and scored:
            with TRACER.span("graph_boost") as span:
                seeds = self.graph.entities_in(search_text)
                if seeds:
                    activation = self.graph.spreading_activation(seeds, hops=2)
                    boosts = self.graph.points_for_entities(activation)
                    if boosts:
                        peak = max(boosts.values()) or 1.0
                        scored = [
                            (point, final + 0.12 * (boosts.get(point.id, 0.0) / peak),
                             {**breakdown, "graph_boost": round(boosts.get(point.id, 0.0) / peak, 4)},
                             rerank)
                            for point, final, breakdown, rerank in scored
                        ]
                        scored.sort(key=lambda row: -row[1])
                    result.graph_context = {
                        "seed_entities": seeds[:6],
                        "activated": sorted(
                            ({"entity": e, "activation": round(v, 3)}
                             for e, v in activation.items() if e not in seeds),
                            key=lambda row: -row["activation"])[:6],
                        "boosted_points": len(boosts),
                    }
                span.set(seeds=len(seeds))

        # 5b. the on-device adapter, trained from this node's own feedback
        adapter_deltas: dict[str, float] = {}
        if self.adapter is not None and len(scored) > 1 and allowed("adapter"):
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

        # 5c. diversity: five phrasings of one incident is one answer, not five
        if self.diversity is not None and allowed("diversity") and len(scored) > k:
            with TRACER.span("diversity") as span:
                candidates_for_mmr = [
                    (point.id, final, np.asarray(point.dense, dtype=np.float32))
                    for point, final, _, _ in scored if point.dense
                ]
                chosen, report = self.diversity.select(vector, candidates_for_mmr, k)
                if chosen:
                    order = {pid: rank for rank, (pid, _) in enumerate(chosen)}
                    scored = (sorted([row for row in scored if row[0].id in order],
                                     key=lambda row: order[row[0].id])
                              + [row for row in scored if row[0].id not in order])
                    result.diversity = report.as_dict()
                span.set(**report.as_dict())

        top = scored[:k]

        # 5d. calibrated confidence: a prediction set with a coverage
        #     guarantee, or an honest abstention
        if self.conformal is not None:
            with TRACER.span("conformal") as span:
                prediction = self.conformal.predict(
                    [(point.id, final) for point, final, _, _ in scored], max_set=max(k, 10))
                result.confidence = prediction.as_dict()
                span.set(set_size=len(prediction.prediction_set), abstained=prediction.abstained)

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
        # Report to the SLO manager here, where *every* query passes, rather
        # than at the HTTP layer where only some do. Observing in the
        # transport left the degradation ladder blind to the WebSocket path,
        # the agent, the mesh and every internal retrieval: a sustained
        # overload measured p99 at 1,850 ms against a 150 ms target while the
        # burn rate sat at exactly 0.0 and the ladder never left FULL, because
        # its latency window was empty. A controller cannot shed load it
        # cannot see.
        if self.slo is not None:
            self.slo.observe(result.latency_ms, ok=True)
        if self.energy is not None:
            # Attributed at the end, from CPU time actually consumed, so a
            # query that waited on the micro-batcher is not billed for it.
            self.energy.sample("query", wall_seconds=result.latency_ms / 1000.0,
                               cpu_seconds=max(cpu_used(), 0.0))
        # Cacheable under the key computed above, which already accounts for
        # the filter. Writes bump the cache epoch, so a stored answer cannot
        # outlive the corpus it was drawn from.
        payload = result.as_dict()
        self.cache.put(vector, payload, namespace)
        self.cache.put_exact(search_text, payload, namespace)
        if not result.results and inferred and _relaxable:
            return await self._relax(
                result, inferred, query, k, asked_collection, mode, explain,
                allow_escalation, asked_filters, tenant_id, result.stages)

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
            "relaxed_inferences": self.relaxed_inferences,
            "answered_before_embedding": self.queries_from_cache,
            "cache": self.cache.snapshot(),
            "latency_ms": hist.snapshot() if hist else {},
        }
        if self.understanding is not None:
            out["understanding"] = self.understanding.snapshot()
        if self.adapter is not None:
            out["adapter"] = self.adapter.snapshot()
        return out
