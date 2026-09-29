"""Vector store abstraction.

`QdrantEdgeStore` binds to Qdrant Edge in-process when the wheel is present.
`NativeStore` is the pure-NumPy tiered implementation used otherwise, so the
node boots and demos on any machine with identical semantics. Both satisfy the
same protocol, and the active backend is reported in /health rather than
silently swapped.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .ann import CostModel
from .filters import Filter
from .index import CollectionIndex
from .planner import PlanKind, QueryPlan
from .qdrant_native import DENSE, HybridPath, NativePlan
from .schema import MemoryPoint, Tier

# Imported for the bake-off only: the native path fuses in the engine, so the
# control needs the interpreter's fusion to compare against.
from ..retrieval.fusion import reciprocal_rank_fusion


class VectorStore(Protocol):
    backend: str

    def upsert(self, point: MemoryPoint) -> None: ...
    def delete(self, point_id: str) -> None: ...
    def move(self, point_id: str, tier: Tier) -> bool: ...
    def search_dense(self, collection: str, query: np.ndarray, k: int,
                     allow: set[str] | None = None) -> list[tuple[str, float]]: ...
    def search_sparse(self, collection: str, sparse: dict[int, float], k: int,
                      allow: set[str] | None = None) -> list[tuple[str, float]]: ...
    def counts(self) -> dict[str, dict[str, int]]: ...


class NativeStore:
    """Tiered store — adaptive ANN, on-disk cold tier, no external service."""

    backend = "native-tiered"

    def __init__(self, dim: int, collections: tuple[str, ...], data_dir: str | None = None) -> None:
        self.dim = dim
        self.data_dir = Path(data_dir) if data_dir else None
        self.cost = CostModel().calibrate(dim=dim)
        self.indexes: dict[str, CollectionIndex] = {
            c: CollectionIndex(dim, self.cost, c, self.data_dir) for c in collections
        }

    def _index(self, collection: str) -> CollectionIndex:
        if collection not in self.indexes:
            self.indexes[collection] = CollectionIndex(self.dim, self.cost, collection, self.data_dir)
        return self.indexes[collection]

    @staticmethod
    def _payload_of(point: MemoryPoint) -> dict[str, Any]:
        """Flatten the fields the planner is allowed to reason about.

        `tenant_id` is here because it was missing: the payload written to
        Qdrant carried no tenant marker at all, so the payload index the store
        asks Qdrant to build for `tenant_id` indexed a field that was never
        written, and nothing reading the stored payload could tell whose memory
        it was looking at.

        This is defence in depth, not the boundary. Tenancy is enforced by
        intersecting the visible id set in `MemoryStore.visible()`, before any
        of this is consulted, and that stays where it is.
        """
        return {
            "collection": point.collection, "sensitivity": point.sensitivity.value,
            "sync_class": point.sync_class.value, "model_version": point.model_version,
            "device_id": point.device_id, "ts": point.created_at,
            "confidence": point.confidence, "stale": point.stale, "pinned": point.pinned,
            "source": point.source or "", "tenant_id": point.tenant_id,
            **point.payload,
        }

    def upsert(self, point: MemoryPoint) -> None:
        vector = np.asarray(point.dense, dtype=np.float32)
        if vector.shape[0] != self.dim:
            raise ValueError(f"dim mismatch: {vector.shape[0]} != {self.dim}")
        self._index(point.collection).upsert(point.id, vector, point.sparse, point.tier,
                                             self._payload_of(point))

    def delete(self, point_id: str) -> None:
        for index in self.indexes.values():
            index.remove(point_id)

    def move(self, point_id: str, tier: Tier) -> bool:
        return any(index.move(point_id, tier) for index in self.indexes.values())

    # -- planning ---------------------------------------------------------

    def plan(self, collection: str, spec: Filter, k: int) -> tuple[QueryPlan, set[str] | None]:
        """Plan once per query; the same allow-set then drives both spaces.

        Across a `*` search each collection plans separately, and the results
        have to be *aggregated*, not competed. A collection holding none of
        the matching points produces a correct EMPTY plan for itself — letting
        that plan win would answer the whole query with nothing.
        """
        targets = list(self.indexes.values()) if collection == "*" else [self._index(collection)]
        plans = [(index, index.plan(spec, k)) for index in targets]
        if not plans:
            return QueryPlan(PlanKind.EMPTY, 0.0, 0, 0, 0.0, "no such collection", allow=set()), set()

        resolvable = [(ix, p) for ix, p in plans if p.allow is not None]
        matched = [(ix, p) for ix, p in plans if p.kind is not PlanKind.EMPTY]

        if not matched:
            merged = QueryPlan(PlanKind.EMPTY, 0.0, 0, 0,
                               sum(p.cost_estimate for _, p in plans),
                               "filter matches nothing in any collection", allow=set())
            return merged, set()

        # every collection could resolve its own ids → one union pre-filter
        if len(resolvable) == len(plans):
            allow: set[str] = set()
            for _, plan in resolvable:
                allow |= (plan.allow or set())
            corpus = sum(len(ix) for ix in targets)
            merged = QueryPlan(
                PlanKind.PRE_FILTER,
                selectivity=(len(allow) / corpus) if corpus else 0.0,
                estimated_matches=len(allow), fetch_k=k,
                cost_estimate=sum(p.cost_estimate for _, p in resolvable),
                reason=(f"filter resolves to {len(allow)} ids across "
                        f"{len(matched)} collection(s) — exact scan of that subset"),
                allow=allow,
            )
            return merged, allow

        best = max(matched, key=lambda row: row[1].estimated_matches)[1]
        return best, None

    def search_dense(self, collection: str, query: np.ndarray, k: int,
                     allow: set[str] | None = None) -> list[tuple[str, float]]:
        if collection == "*":
            merged: list[tuple[str, float]] = []
            for index in self.indexes.values():
                merged.extend(index.search_dense(query, k, allow=allow))
            merged.sort(key=lambda x: -x[1])
            return merged[:k]
        return self._index(collection).search_dense(query, k, allow=allow)

    def search_sparse(self, collection: str, sparse: dict[int, float], k: int,
                      allow: set[str] | None = None) -> list[tuple[str, float]]:
        if collection == "*":
            merged: list[tuple[str, float]] = []
            for index in self.indexes.values():
                merged.extend(index.search_sparse(sparse, k, allow=allow))
            merged.sort(key=lambda x: -x[1])
            return merged[:k]
        return self._index(collection).search_sparse(sparse, k, allow=allow)

    def close(self) -> None:
        """No external handles to release for the internal store."""

    def migrate_pending(self) -> list[dict[str, Any]]:
        """Perform every deferred index rebuild; returns what it did."""
        done = []
        for name, index in self.indexes.items():
            result = index.migrate_pending()
            if result:
                done.append({"collection": name, **result})
        return done

    def index_report(self) -> dict[str, Any]:
        return {"cost_model": self.cost.as_dict(),
                "collections": {name: ix.snapshot() for name, ix in self.indexes.items()}}

    def counts(self) -> dict[str, dict[str, int]]:
        return {name: index.counts() for name, index in self.indexes.items()}

    @property
    def rescored(self) -> int:
        return sum(i.rescored for i in self.indexes.values())

    @property
    def resident_bytes(self) -> int:
        return sum(i.storage.snapshot()["resident_bytes"] for i in self.indexes.values())


class StorageLocked(RuntimeError):
    """Another process already holds this data directory."""


class QdrantStore(NativeStore):
    """Qdrant as the system of record *and* as the query engine.

    Every collection is provisioned with the shape the whole hybrid pipeline
    needs: a named `dense` vector, a `lex` sparse vector, and — when the late
    budget is paid for — a `late` multivector with a MAX_SIM comparator. That
    means one `query_points` call can do dense recall, sparse recall, rank
    fusion and late-interaction rerank inside the engine, which is what
    `search_native()` does. See `qdrant_native.HybridPath`.

    The local adaptive index stays, and it is not decoration. Two reasons.
    It is the control: `bake_off()` runs the same queries down both paths and
    compares neighbours, ranking and latency, so "Qdrant is faster here" is a
    measurement rather than a slogan. And it is the answer for what the engine
    refuses — a filter with no faithful Qdrant form, a collection still on the
    old single-vector schema — where the native path *declines* instead of
    quietly answering a different question.

    Which path a query took is reported per query, and the active backend
    verbatim in `/health`: `qdrant-server` when a URL is configured,
    `qdrant-local` when embedded.
    """

    def __init__(self, dim: int, collections: tuple[str, ...], path: str,
                 url: str | None = None, api_key: str | None = None,
                 late_tokens: int = 0) -> None:
        super().__init__(dim, collections, path)
        from qdrant_client import QdrantClient

        self.url = url
        if url:
            self.client = QdrantClient(url=url, api_key=api_key, timeout=10)
            self.backend = "qdrant-server"
        else:
            storage = Path(path) / "qdrant"
            try:
                self.client = QdrantClient(path=str(storage))
            except Exception as exc:
                # Embedded Qdrant is single-writer by design. Two node processes
                # on one directory would corrupt it, so the lock is correct —
                # but the raw portalocker BlockingIOError tells an operator
                # nothing about what to do next.
                if "lock" in str(exc).lower() or isinstance(exc, BlockingIOError):
                    raise StorageLocked(
                        f"another AegisEdge process already holds {storage}. "
                        f"Embedded Qdrant allows one writer: stop the other node, "
                        f"or point this one at its own AEGIS_DATA_DIR."
                    ) from exc
                raise
            self.backend = "qdrant-local"

        self.collections = collections
        self.hybrid = HybridPath(self.client, dim, collections, late_tokens=late_tokens,
                                 server=bool(url))
        self.provisioning = self.hybrid.provision()
        self.late_provider = None               # set by the node when it has a reranker
        self.upserts = 0
        self.remote_searches = 0
        self.native_searches = 0
        self.native_refusals = 0
        self.last_plan: NativePlan | None = None

    @staticmethod
    def _qdrant_id(point_id: str) -> str:
        """Qdrant ids are UUIDs or unsigned ints; ours are ULIDs."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, point_id))

    def upsert(self, point: MemoryPoint) -> None:
        super().upsert(point)
        from qdrant_client.models import PointStruct

        payload = {**self._payload_of(point), "aegis_id": point.id, "text": point.text}
        if self.hybrid.native(point.collection):
            late = None
            if self.late_provider is not None and self.hybrid.late_tokens:
                try:
                    late = self.late_provider(point.text)
                except Exception:
                    late = None                 # a missing residual degrades rank, not writes
            vector = self.hybrid.vector_of(point.collection, point.dense,
                                           point.sparse, late)
        else:
            vector = point.dense_list()         # a collection still on the old schema
        self.client.upsert(
            collection_name=point.collection,
            points=[PointStruct(id=self._qdrant_id(point.id), vector=vector, payload=payload)],
        )
        self.upserts += 1

    def delete(self, point_id: str) -> None:
        super().delete(point_id)
        from qdrant_client.models import PointIdsList

        for name in self.collections:
            try:
                self.client.delete(collection_name=name,
                                   points_selector=PointIdsList(points=[self._qdrant_id(point_id)]))
            except Exception:
                continue

    def search_remote(self, collection: str, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Ask Qdrant itself — the path a server deployment would take."""
        self.remote_searches += 1
        targets = self.collections if collection == "*" else (collection,)
        merged: list[tuple[str, float]] = []
        for name in targets:
            try:
                hits = self.client.query_points(
                    collection_name=name,
                    query=np.asarray(query, dtype=np.float32).tolist(),
                    using=DENSE if self.hybrid.native(name) else None,
                    limit=k, with_payload=True).points
            except Exception:
                continue
            merged.extend((h.payload.get("aegis_id", str(h.id)), float(h.score)) for h in hits)
        merged.sort(key=lambda row: -row[1])
        return merged[:k]

    def migrate_schema(self, points: list[MemoryPoint], force: bool = False) -> dict[str, Any]:
        """Move a legacy single-vector collection onto the hybrid schema.

        Recreating a collection deletes it first, which is only safe because
        Qdrant is not the only copy: every memory is also in the write-ahead
        log and the local index, and `points` is that copy being handed back.
        The guard is the part that matters — if Qdrant holds more points for a
        collection than the caller is offering to rewrite, the difference is
        data only Qdrant has, and this refuses rather than deleting it.

        `force` overrides that refusal, and says in the report that it did.
        """
        from qdrant_client.models import SparseVectorParams

        report: dict[str, Any] = {"migrated": [], "skipped": [], "refused": [], "points": 0}
        by_collection: dict[str, list[MemoryPoint]] = {}
        for point in points:
            by_collection.setdefault(point.collection, []).append(point)

        for name in self.collections:
            schema = self.hybrid.schema.get(name, "")
            if schema.startswith("hybrid") and not self.hybrid.wants_late(name):
                report["skipped"].append({"collection": name, "reason": "already hybrid"})
                continue
            if self.hybrid.wants_late(name):
                # Configured for late interaction after this collection was
                # created. Same recreate, different reason, and it has to be
                # said out loud rather than reported as a legacy upgrade.
                report.setdefault("reasons", {})[name] = (
                    "adding the late-interaction vector this collection was created without")
            mine = by_collection.get(name, [])
            try:
                held = int(getattr(self.client.get_collection(name), "points_count", 0) or 0)
            except Exception:
                held = 0
            if held > len(mine) and not force:
                report["refused"].append({
                    "collection": name, "in_qdrant": held, "offered": len(mine),
                    "reason": ("Qdrant holds points this migration was not given; "
                               "recreating would delete them"),
                })
                continue
            self.client.delete_collection(name)
            self.client.create_collection(
                collection_name=name,
                vectors_config=self.hybrid._vectors_config(),
                sparse_vectors_config={"lex": SparseVectorParams()},
            )
            self.hybrid.schema[name] = ("hybrid-late" if self.hybrid.late_tokens else "hybrid")
            for point in mine:
                self.upsert(point)
            report["migrated"].append({"collection": name, "rewritten": len(mine),
                                       "was_holding": held, "forced": bool(force and held > len(mine))})
            report["points"] += len(mine)
        self.hybrid._index_payloads()
        report["schema"] = dict(self.hybrid.schema)
        return report

    def search_native(self, collection: str, dense: np.ndarray, sparse: dict[int, float],
                      k: int, *, fetch: int | None = None, spec: Filter | None = None,
                      late: np.ndarray | None = None, mode: str = "hybrid",
                      ) -> tuple[list[tuple[str, float]], NativePlan]:
        """Run the whole hybrid pipeline in the engine, in one call.

        On a refusal the hits are empty and `plan.fell_back` says why; the
        caller must read that flag rather than the list, because "the engine
        declined" and "there are no matching memories" are different answers
        and conflating them is how a filtered search quietly goes wrong.
        """
        hits, plan = self.hybrid.query(collection, dense, sparse, k, fetch=fetch,
                                       spec=spec, late=late, mode=mode)
        plan.engine = self.backend
        self.last_plan = plan
        if plan.fell_back:
            self.native_refusals += 1
        else:
            self.native_searches += 1
        return hits, plan

    def bake_off(self, collection: str, dense: np.ndarray, sparse: dict[int, float],
                 k: int = 5, *, spec: Filter | None = None) -> dict[str, Any]:
        """The same query down both paths, with the disagreement named.

        Rank-biased overlap is reported alongside plain overlap because the two
        failure modes are different: the paths can retrieve the same set and
        order it differently, and for a top-k answer the order is the product.
        """
        t0 = time.perf_counter()
        allow = None
        if spec is not None and not spec.is_empty():
            _, allow = self.plan(collection, spec, k)
        local_hits = self.search_dense(collection, dense, k, allow=allow)
        local_sparse = self.search_sparse(collection, sparse, k, allow=allow)
        local_ms = (time.perf_counter() - t0) * 1000

        native_hits, plan = self.search_native(collection, dense, sparse, k, spec=spec)
        local = [hit.point_id for hit in reciprocal_rank_fusion(local_hits, local_sparse)][:k]
        native = [pid for pid, _ in native_hits]
        shared = set(local) & set(native)
        agreement = len(shared) / max(len(local) or 1, 1)
        # rank agreement: average, over prefixes, of how much of the local
        # top-i the engine also put in its top-i
        prefixes = [len(set(local[:i]) & set(native[:i])) / i
                    for i in range(1, min(len(local), len(native)) + 1)]
        return {
            "k": k, "local": local, "native": native,
            "overlap": round(agreement, 3),
            "rank_agreement": round(sum(prefixes) / len(prefixes), 3) if prefixes else 0.0,
            "local_ms": round(local_ms, 3), "native_ms": round(plan.total_ms, 3),
            "speedup": round(local_ms / plan.total_ms, 2) if plan.total_ms > 0 else None,
            "plan": plan.as_dict(), "backend": self.backend,
        }

    def verify_agreement(self, collection: str, query: np.ndarray, k: int = 5) -> dict[str, Any]:
        """Do the local index and Qdrant return the same neighbours?"""
        local = [pid for pid, _ in self.search_dense(collection, query, k)]
        remote = [pid for pid, _ in self.search_remote(collection, query, k)]
        overlap = len(set(local) & set(remote)) / max(len(local) or 1, 1)
        return {"local": local, "remote": remote, "overlap": round(overlap, 3),
                "agree": overlap >= 0.8, "backend": self.backend}

    def counts(self) -> dict[str, dict[str, int]]:
        return super().counts()

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass

    def index_report(self) -> dict[str, Any]:
        report = super().index_report()
        report["qdrant"] = {
            "backend": self.backend, "url": self.url,
            "upserts": self.upserts, "remote_searches": self.remote_searches,
            "native_searches": self.native_searches,
            "native_refusals": self.native_refusals,
            "hybrid": self.hybrid.snapshot(),
            "provisioning": self.provisioning,
            "collections": {
                name: getattr(self.client.get_collection(name), "points_count", None)
                for name in self.collections
            },
        }
        return report


def build_store(dim: int, collections: tuple[str, ...], path: str,
                url: str | None = None, api_key: str | None = None,
                required: bool = True, late_tokens: int = 0) -> VectorStore:
    """Open the vector store.

    Qdrant is a hard requirement by default: a node that silently falls back to
    an internal store while claiming to be Qdrant-backed is telling its
    operator something untrue about where their data lives. Set
    `AEGIS_REQUIRE_QDRANT=0` to allow the internal store explicitly.
    """
    try:
        return QdrantStore(dim, collections, path, url, api_key, late_tokens)
    except Exception as exc:
        if isinstance(exc, StorageLocked):
            raise
        if required:
            raise RuntimeError(
                f"Qdrant is required but could not be opened ({exc}). "
                f"Install `qdrant-client`, or set AEGIS_REQUIRE_QDRANT=0 to run on the "
                f"internal store."
            ) from exc
        return NativeStore(dim, collections, path)
