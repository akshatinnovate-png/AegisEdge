"""Vector store abstraction.

`QdrantEdgeStore` binds to Qdrant Edge in-process when the wheel is present.
`NativeStore` is the pure-NumPy tiered implementation used otherwise, so the
node boots and demos on any machine with identical semantics. Both satisfy the
same protocol, and the active backend is reported in /health rather than
silently swapped.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .ann import CostModel
from .filters import Filter
from .index import CollectionIndex
from .planner import PlanKind, QueryPlan
from .schema import MemoryPoint, Tier


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
        """Flatten the fields the planner is allowed to reason about."""
        return {
            "collection": point.collection, "sensitivity": point.sensitivity.value,
            "sync_class": point.sync_class.value, "model_version": point.model_version,
            "device_id": point.device_id, "ts": point.created_at,
            "confidence": point.confidence, "stale": point.stale, "pinned": point.pinned,
            "source": point.source or "", **point.payload,
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
    """Qdrant as the system of record, the adaptive index as the query path.

    Two honest choices are encoded here.

    First, Qdrant owns persistence and the collection API — real `PointStruct`
    upserts, real payloads, real filters — so the same adapter that runs
    against an embedded instance runs against a Qdrant Server by changing one
    setting.

    Second, the hot query path stays on the local adaptive index. The embedded
    client's local mode is a pure-Python implementation intended for
    development; this node's own index has a calibrated HNSW and an OPQ/IVF-PQ
    tier, so routing hot queries through local mode would be slower and less
    accurate. Against a real server the engine is Rust and that trade flips —
    `search_remote()` exists for exactly that, and `verify_agreement()` proves
    the two paths return the same neighbours rather than asking anyone to
    assume it.

    The active backend is reported verbatim in `/health`: `qdrant-server` when
    a URL is configured, `qdrant-local` when embedded.
    """

    def __init__(self, dim: int, collections: tuple[str, ...], path: str,
                 url: str | None = None, api_key: str | None = None) -> None:
        super().__init__(dim, collections, path)
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

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
        for name in collections:
            if not self.client.collection_exists(name):
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
        # Payload indexes are a server feature; local mode filters without them.
        if url:
            for name in collections:
                for field_name, schema in (("ts", "float"), ("sensitivity", "keyword"),
                                           ("tenant_id", "keyword"), ("model_version", "keyword")):
                    try:
                        self.client.create_payload_index(name, field_name=field_name,
                                                         field_schema=schema)
                    except Exception:
                        pass                    # already present
        self.upserts = 0
        self.remote_searches = 0

    @staticmethod
    def _qdrant_id(point_id: str) -> str:
        """Qdrant ids are UUIDs or unsigned ints; ours are ULIDs."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, point_id))

    def upsert(self, point: MemoryPoint) -> None:
        super().upsert(point)
        from qdrant_client.models import PointStruct

        payload = {**self._payload_of(point), "aegis_id": point.id, "text": point.text}
        self.client.upsert(
            collection_name=point.collection,
            points=[PointStruct(id=self._qdrant_id(point.id),
                                vector=point.dense_list(), payload=payload)],
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
                hits = self.client.query_points(collection_name=name,
                                                query=list(np.asarray(query, dtype=np.float32)),
                                                limit=k, with_payload=True).points
            except Exception:
                continue
            merged.extend((h.payload.get("aegis_id", str(h.id)), float(h.score)) for h in hits)
        merged.sort(key=lambda row: -row[1])
        return merged[:k]

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
            "collections": {
                name: getattr(self.client.get_collection(name), "points_count", None)
                for name in self.collections
            },
        }
        return report


def build_store(dim: int, collections: tuple[str, ...], path: str,
                url: str | None = None, api_key: str | None = None,
                required: bool = True) -> VectorStore:
    """Open the vector store.

    Qdrant is a hard requirement by default: a node that silently falls back to
    an internal store while claiming to be Qdrant-backed is telling its
    operator something untrue about where their data lives. Set
    `AEGIS_REQUIRE_QDRANT=0` to allow the internal store explicitly.
    """
    try:
        return QdrantStore(dim, collections, path, url, api_key)
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
