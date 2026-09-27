"""The Qdrant-native hybrid query path.

The node's retrieval is four stages: dense recall, sparse recall, reciprocal
rank fusion, late-interaction rerank. Every one of those is a thing Qdrant
does natively, and doing them in Python instead means four round trips, four
result sets materialised in the interpreter, and a fusion that the engine
could have done while the postings were still warm.

This module expresses the whole pipeline as *one* `query_points` call:

    prefetch(dense, limit=fetch)  ┐
    prefetch(lex,   limit=fetch)  ┴─→ FusionQuery(RRF) ─→ query(late, MaxSim)

Three things make that safe to switch on rather than merely impressive.

`translate()` refuses. A filter this module cannot express as a Qdrant filter
returns `None` and the caller stays on the Python path — a native search that
silently dropped a restriction would answer one tenant's query with another
tenant's memories, so an untranslatable filter is a *refusal*, never a
widening.

`plan()` measures. The returned plan names each stage, its limit, whether it
ran in the engine or the interpreter, and how long the single call took, so
the query panel shows what actually happened rather than an architecture
diagram.

Late interaction is bounded and priced. A full ColBERT residual for one
memory is `tokens × dim` floats — at 128 tokens and 384 dimensions that is
196 KB, ninety times what a whole memory costs today. So the late vector is
pruned to a token budget and the cost is reported per collection; it stays off
until someone looks at that number and decides to pay it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .filters import Condition, Filter, Op

DENSE = "dense"
SPARSE = "lex"
LATE = "late"

# Fields worth an index on a server. Local mode warns that indexes do nothing
# there, and it is right — see `payload_indexes` in the provisioning report.
INDEXED_FIELDS: tuple[tuple[str, str], ...] = (
    ("aegis_id", "keyword"), ("collection", "keyword"), ("sensitivity", "keyword"),
    ("sync_class", "keyword"), ("model_version", "keyword"), ("device_id", "keyword"),
    ("tenant_id", "keyword"), ("source", "keyword"),
    ("ts", "float"), ("confidence", "float"),
    ("stale", "bool"), ("pinned", "bool"),
)


class Untranslatable(ValueError):
    """This filter cannot be expressed to Qdrant without changing its meaning."""


def _condition(condition: Condition) -> Any:
    from qdrant_client import models as qm

    key, value = condition.field, condition.value
    if condition.op is Op.EQ:
        if isinstance(value, bool):
            return qm.FieldCondition(key=key, match=qm.MatchValue(value=value))
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return qm.FieldCondition(key=key, range=qm.Range(gte=value, lte=value))
        return qm.FieldCondition(key=key, match=qm.MatchValue(value=str(value)))
    if condition.op is Op.IN:
        values = list(value or ())
        if not values:
            raise Untranslatable("IN over an empty set")
        if all(isinstance(v, int) and not isinstance(v, bool) for v in values):
            return qm.FieldCondition(key=key, match=qm.MatchAny(any=values))
        return qm.FieldCondition(key=key, match=qm.MatchAny(any=[str(v) for v in values]))
    if condition.op in (Op.LT, Op.LTE, Op.GT, Op.GTE):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise Untranslatable(f"{condition.op.value} on a non-numeric bound")
        bound = {Op.LT: "lt", Op.LTE: "lte", Op.GT: "gt", Op.GTE: "gte"}[condition.op]
        return qm.FieldCondition(key=key, range=qm.Range(**{bound: float(value)}))
    if condition.op is Op.RANGE:
        try:
            low, high = value
        except (TypeError, ValueError) as exc:
            raise Untranslatable("RANGE without two bounds") from exc
        return qm.FieldCondition(key=key, range=qm.Range(gte=float(low), lte=float(high)))
    # NE is not a condition in Qdrant's algebra — it is a negated match, so
    # `translate()` moves it across the tree rather than encoding it here.
    # EXISTS and CONTAINS have no form that means the same thing: Qdrant's
    # is-null and is-empty are not our "present", and our CONTAINS reads both
    # substrings and list membership. Both are refusals, not approximations.
    raise Untranslatable(f"no faithful Qdrant form for {condition.op.value}")


def translate(spec: Filter | None) -> Any | None:
    """A Qdrant filter with the same meaning, or None if there isn't one.

    `None` means *stay on the Python path*. It never means "no filter": the
    caller must not treat an untranslatable restriction as an absent one.
    """
    if spec is None or spec.is_empty():
        return None
    from qdrant_client import models as qm

    def flip(condition: Condition) -> Condition:
        """`field != v` as the positive condition Qdrant negates."""
        return Condition(condition.field, Op.EQ, condition.value)

    must, should, must_not = [], [], []
    try:
        for condition in spec.must:
            # AND(x != v) is exactly NOT(x == v), which crosses the tree.
            (must_not if condition.op is Op.NE else must).append(
                _condition(flip(condition) if condition.op is Op.NE else condition))
        for condition in spec.must_not:
            if condition.op is Op.NE:
                # NOT(x != v) is AND(x == v) only while x is present, and our
                # payloads are heterogeneous enough that it sometimes is not.
                raise Untranslatable("must_not over NE")
            must_not.append(_condition(condition))
        for condition in spec.should:
            if condition.op is Op.NE:
                # OR of a negation has no single-condition form; a nested
                # should-clause of Filters would express it, and guessing is
                # how a filter quietly stops meaning what it said.
                raise Untranslatable("should over NE")
            should.append(_condition(condition))
    except Untranslatable:
        return None
    return qm.Filter(must=must or None, should=should or None, must_not=must_not or None)


def translatable(spec: Filter | None) -> bool:
    return spec is None or spec.is_empty() or translate(spec) is not None


@dataclass(slots=True)
class Stage:
    name: str
    where: str                     # "engine" or "interpreter"
    detail: str
    limit: int = 0
    hits: int = 0
    ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "where": self.where, "detail": self.detail,
                "limit": self.limit, "hits": self.hits, "ms": round(self.ms, 3)}


@dataclass
class NativePlan:
    """What the engine was asked to do, and what it cost."""
    engine: str = "qdrant"
    calls: int = 0
    round_trips: int = 0
    stages: list[Stage] = field(default_factory=list)
    fusion: str = "rrf"
    rerank: str = "none"
    queries_in_call: int = 0
    # point_id -> {dense_rank, sparse_rank, dense_score, sparse_score}. The
    # engine fuses internally, so which space matched would otherwise be lost;
    # it is recovered in the same round trip rather than with extra ones.
    attribution: dict[str, dict[str, Any]] = field(default_factory=dict)
    filtered: bool = False
    prefilter: bool = False
    total_ms: float = 0.0
    itemised: bool = False     # one call in, one timing out
    fell_back: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"engine": self.engine, "calls": self.calls, "round_trips": self.round_trips,
                "stages": [s.as_dict() for s in self.stages], "fusion": self.fusion,
                "rerank": self.rerank, "filtered": self.filtered,
                "prefilter": self.prefilter, "total_ms": round(self.total_ms, 3),
                "itemised": self.itemised, "queries_in_call": self.queries_in_call,
                "attributed": len(self.attribution), "fell_back": self.fell_back}


def prune_late(vectors: np.ndarray, budget: int) -> np.ndarray:
    """Keep the `budget` most distinctive token vectors, in reading order.

    MaxSim takes each query token's best document token, so a document's value
    is carried by its unusual tokens; the ones closest to the document's own
    centroid are the ones any passage would have supplied. Dropping those
    costs the least, and keeping reading order keeps the residual legible when
    someone dumps it.
    """
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0:
        return np.zeros((0, array.shape[1] if array.ndim == 2 else 0), dtype=np.float32)
    if budget <= 0 or array.shape[0] <= budget:
        return np.ascontiguousarray(array)
    centroid = array.mean(axis=0)
    norms = np.linalg.norm(array, axis=1) * (np.linalg.norm(centroid) or 1.0)
    affinity = (array @ centroid) / np.where(norms == 0, 1.0, norms)
    keep = np.sort(np.argsort(affinity)[:budget])
    return np.ascontiguousarray(array[keep])


class HybridPath:
    """Provisions and queries the collections that hold the native pipeline.

    It owns no client of its own: `QdrantStore` passes the one it opened, so
    embedded and server deployments take exactly the same code path and the
    only difference is which engine executes the plan.
    """

    def __init__(self, client, dim: int, collections: tuple[str, ...], *,
                 late_tokens: int = 0, quantize_late: bool = True,
                 server: bool = False) -> None:
        self.client = client
        self.server = server
        self.dim = dim
        self.collections = tuple(collections)
        self.late_tokens = int(late_tokens)
        self.quantize_late = quantize_late
        self.schema: dict[str, str] = {}
        self.payload_indexes: dict[str, str] = {}
        self.queries = 0
        self.refusals = 0
        self.late_points = 0

    # -- provisioning -----------------------------------------------------

    def _vectors_config(self) -> dict[str, Any]:
        from qdrant_client import models as qm

        config: dict[str, Any] = {
            DENSE: qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE),
        }
        if self.late_tokens > 0:
            config[LATE] = qm.VectorParams(
                size=self.dim, distance=qm.Distance.COSINE,
                multivector_config=qm.MultiVectorConfig(
                    comparator=qm.MultiVectorComparator.MAX_SIM),
                # A multivector is only ever reached through a prefetch, so an
                # HNSW graph over it would be built and never traversed.
                hnsw_config=qm.HnswConfigDiff(m=0),
                quantization_config=(qm.BinaryQuantization(
                    binary=qm.BinaryQuantizationConfig(always_ram=True))
                    if self.quantize_late else None),
            )
        return config

    def provision(self) -> dict[str, Any]:
        """Create what is missing; classify what is already there.

        A collection created by an older build holds one unnamed vector. That
        is not something to "fix" by recreating it — the recreate would delete
        the operator's memories. It is recorded as `legacy`, and every legacy
        collection keeps the Python path.
        """
        from qdrant_client import models as qm

        created = []
        for name in self.collections:
            if self.client.collection_exists(name):
                self.schema[name] = self._classify(name)
                continue
            self.client.create_collection(
                collection_name=name,
                vectors_config=self._vectors_config(),
                sparse_vectors_config={SPARSE: qm.SparseVectorParams()},
            )
            self.schema[name] = "hybrid-late" if self.late_tokens > 0 else "hybrid"
            created.append(name)
        self._index_payloads()
        return {"created": created, "schema": dict(self.schema),
                "payload_indexes": dict(self.payload_indexes),
                "late_tokens": self.late_tokens}

    def _classify(self, name: str) -> str:
        try:
            info = self.client.get_collection(name)
            vectors = info.config.params.vectors
            sparse = info.config.params.sparse_vectors or {}
        except Exception:
            return "unknown"
        if not isinstance(vectors, dict) or DENSE not in vectors:
            return "legacy"
        if SPARSE not in sparse:
            return "legacy"
        return "hybrid-late" if LATE in vectors else "hybrid"

    def _index_payloads(self) -> None:
        """Ask for the indexes, and record honestly whether they can take.

        Embedded local mode says it plainly — "payload indexes have no effect
        in the local Qdrant" — so recording `ignored-by-local-mode` there is
        not pessimism, it is the client's own answer. The difference between a
        system that knows which of its optimisations are live and one that
        assumes they all are shows up the first time a filtered query is slow.
        """
        for name in self.collections:
            if not self.schema.get(name, "").startswith("hybrid"):
                continue
            for field_name, schema in INDEXED_FIELDS:
                key = f"{name}.{field_name}"
                if not self.server:
                    self.payload_indexes[key] = "ignored-by-local-mode"
                    continue
                try:
                    self.client.create_payload_index(name, field_name=field_name,
                                                     field_schema=schema)
                    self.payload_indexes[key] = schema
                except Exception as exc:                 # already present, or refused
                    self.payload_indexes[key] = f"skipped: {type(exc).__name__}"

    def native(self, collection: str) -> bool:
        if collection == "*":
            return bool(self.collections) and all(
                self.schema.get(c, "").startswith("hybrid") for c in self.collections)
        return self.schema.get(collection, "").startswith("hybrid")

    def has_late(self, collection: str) -> bool:
        if not self.late_tokens:
            return False
        names = self.collections if collection == "*" else (collection,)
        return all(self.schema.get(c) == "hybrid-late" for c in names)

    # -- writing ----------------------------------------------------------

    def vector_of(self, dense: np.ndarray, sparse: dict[int, float],
                  late: np.ndarray | None = None) -> dict[str, Any]:
        """The named-vector payload for one point."""
        from qdrant_client import models as qm

        vectors: dict[str, Any] = {DENSE: np.asarray(dense, dtype=np.float32).tolist()}
        if sparse:
            # Qdrant requires unique indices; our hashed term space can collide,
            # and the larger weight is the one that survives.
            merged: dict[int, float] = {}
            for term, weight in sparse.items():
                key = int(term)
                if weight > merged.get(key, float("-inf")):
                    merged[key] = float(weight)
            ordered = sorted(merged.items())
            vectors[SPARSE] = qm.SparseVector(indices=[t for t, _ in ordered],
                                             values=[w for _, w in ordered])
        if late is not None and self.late_tokens > 0:
            pruned = prune_late(late, self.late_tokens)
            if pruned.shape[0]:
                vectors[LATE] = pruned.tolist()
                self.late_points += 1
        return vectors

    def late_bytes(self) -> int:
        """What one late residual costs on the wire, before quantization."""
        return self.late_tokens * self.dim * 4

    # -- reading ----------------------------------------------------------

    def query(self, collection: str, dense: np.ndarray, sparse: dict[int, float], k: int,
              *, fetch: int | None = None, spec: Filter | None = None,
              late: np.ndarray | None = None, mode: str = "hybrid",
              ) -> tuple[list[tuple[str, float]], NativePlan]:
        """One call: recall in both spaces, fuse, and rerank, inside the engine.

        Returns `([], plan)` with `plan.fell_back` set when the query cannot be
        run natively — an untranslatable filter, a legacy collection, a `*`
        search across collections that do not all qualify. The caller reads
        `fell_back` and uses the Python path; it must not read the empty list
        as "no matches".
        """
        from qdrant_client import models as qm

        plan = NativePlan()
        if not self.native(collection):
            plan.fell_back = f"collection schema is {self.schema.get(collection, 'unknown')}"
            self.refusals += 1
            return [], plan

        qdrant_filter = None
        if spec is not None and not spec.is_empty():
            qdrant_filter = translate(spec)
            if qdrant_filter is None:
                plan.fell_back = "filter has no faithful Qdrant form"
                self.refusals += 1
                return [], plan
            plan.filtered = True
            plan.prefilter = True

        fetch = fetch or max(k * 6, 24)
        prefetch: list[Any] = []
        if mode != "sparse":
            prefetch.append(qm.Prefetch(query=np.asarray(dense, dtype=np.float32).tolist(),
                                        using=DENSE, limit=fetch, filter=qdrant_filter))
            plan.stages.append(Stage("dense recall", "engine",
                                     f"HNSW over `{DENSE}`, cosine", limit=fetch))
        if mode != "dense" and sparse:
            vector = self.vector_of(np.zeros(self.dim, dtype=np.float32), sparse)[SPARSE]
            prefetch.append(qm.Prefetch(query=vector, using=SPARSE, limit=fetch,
                                        filter=qdrant_filter))
            plan.stages.append(Stage("sparse recall", "engine",
                                     f"inverted index over `{SPARSE}`, "
                                     f"{len(sparse)} terms", limit=fetch))
        if not prefetch:
            plan.fell_back = f"mode {mode} has nothing to prefetch"
            self.refusals += 1
            return [], plan

        use_late = late is not None and self.has_late(collection) and len(np.asarray(late)) > 0
        targets = self.collections if collection == "*" else (collection,)
        merged: dict[str, float] = {}
        started = time.perf_counter()
        calls = 0
        for name in targets:
            if len(prefetch) == 1 and not use_late:
                fused_request = qm.QueryRequest(
                    query=prefetch[0].query, using=prefetch[0].using,
                    filter=qdrant_filter, limit=k, with_payload=True)
                plan.fusion = "none (single space)"
            elif use_late:
                fused_request = qm.QueryRequest(
                    prefetch=[qm.Prefetch(prefetch=list(prefetch),
                                          query=qm.FusionQuery(fusion=qm.Fusion.RRF),
                                          limit=max(k * 3, 12))],
                    query=prune_late(late, self.late_tokens).tolist(),
                    using=LATE, limit=k, with_payload=True)
            else:
                fused_request = qm.QueryRequest(
                    prefetch=list(prefetch), query=qm.FusionQuery(fusion=qm.Fusion.RRF),
                    limit=k, with_payload=True)

            # The two per-space orderings ride along in the same batch. They are
            # what "matched by dense and sparse" is read from: the engine fuses
            # internally, so without them the answer could not say which half of
            # hybrid retrieval found a memory — and that is one of the few things
            # an operator inspecting a result actually wants to know. A batch is
            # one round trip, so the attribution is free of a second one.
            requests = [fused_request]
            if len(prefetch) > 1:
                requests.extend(
                    qm.QueryRequest(query=p.query, using=p.using, filter=qdrant_filter,
                                    limit=fetch, with_payload=True)
                    for p in prefetch)
            try:
                responses = self.client.query_batch_points(collection_name=name,
                                                           requests=requests)
            except Exception as exc:
                plan.fell_back = f"{type(exc).__name__}: {exc}"
                self.refusals += 1
                return [], plan
            calls += 1
            plan.queries_in_call = len(requests)
            for hit in responses[0].points:
                point_id = (hit.payload or {}).get("aegis_id") or str(hit.id)
                score = float(hit.score)
                if score > merged.get(point_id, float("-inf")):
                    merged[point_id] = score
            for space, response in zip(("dense", "sparse"), responses[1:]):
                for rank, hit in enumerate(response.points, start=1):
                    point_id = (hit.payload or {}).get("aegis_id") or str(hit.id)
                    row = plan.attribution.setdefault(point_id, {})
                    row[f"{space}_rank"] = rank
                    row[f"{space}_score"] = float(hit.score)
        plan.total_ms = (time.perf_counter() - started) * 1000
        plan.calls = calls
        plan.round_trips = calls
        if len(prefetch) > 1:
            plan.stages.append(Stage("fusion", "engine",
                                     "reciprocal rank fusion, engine-side",
                                     limit=max(k * 3, 12) if use_late else k))
        if use_late:
            plan.rerank = f"maxsim ({self.late_tokens} tokens)"
            plan.stages.append(Stage("late interaction", "engine",
                                     f"MaxSim over `{LATE}` multivector", limit=k))
        results = sorted(merged.items(), key=lambda row: -row[1])[:k]
        for stage in plan.stages:
            stage.hits = len(results)
            # Deliberately left at zero. The whole pipeline is one call, and the
            # engine returns one timing for it; splitting that total across the
            # stages evenly would put a number on the panel that nothing
            # measured. `total_ms` is the measurement, and `itemised` says so.
            stage.ms = 0.0
        self.queries += 1
        return results, plan

    def sample(self, collection: str, limit: int = 600) -> tuple[list[str], np.ndarray,
                                                                 list[dict[str, Any]]]:
        """Read points back out of Qdrant, vectors included.

        The console's map is drawn from what the engine holds rather than from
        the node's own copy on purpose: a picture of the corpus that came from
        the process drawing it would agree with itself no matter what Qdrant
        actually stored.
        """
        if not self.native(collection):
            return [], np.zeros((0, self.dim), dtype=np.float32), []
        try:
            rows, _ = self.client.scroll(collection_name=collection, limit=limit,
                                         with_vectors=True, with_payload=True)
        except Exception:
            return [], np.zeros((0, self.dim), dtype=np.float32), []
        ids: list[str] = []
        vectors: list[list[float]] = []
        payloads: list[dict[str, Any]] = []
        for row in rows:
            vector = row.vector.get(DENSE) if isinstance(row.vector, dict) else row.vector
            if vector is None:
                continue
            payload = row.payload or {}
            ids.append(str(payload.get("aegis_id") or row.id))
            vectors.append(list(vector))
            payloads.append(payload)
        if not vectors:
            return [], np.zeros((0, self.dim), dtype=np.float32), []
        return ids, np.asarray(vectors, dtype=np.float32), payloads

    def facet(self, collection: str, key: str, limit: int = 10) -> list[dict[str, Any]]:
        """Payload value counts from the engine, for the inspection UI."""
        try:
            response = self.client.facet(collection_name=collection, key=key, limit=limit)
        except Exception:
            return []
        return [{"value": str(hit.value), "count": int(hit.count)} for hit in response.hits]

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema": dict(self.schema),
            "payload_indexes": dict(self.payload_indexes),
            "vectors": [DENSE, SPARSE] + ([LATE] if self.late_tokens else []),
            "late_tokens": self.late_tokens,
            "late_bytes_per_point": self.late_bytes(),
            "late_points": self.late_points,
            "quantize_late": self.quantize_late,
            "server": self.server,
            "native_queries": self.queries,
            "refusals": self.refusals,
        }
