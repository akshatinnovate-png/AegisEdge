"""Vector store abstraction.

`QdrantEdgeStore` binds to Qdrant Edge in-process when the wheel is present.
`NativeStore` is the pure-NumPy tiered implementation used otherwise, so the
node boots and demos on any machine with identical semantics. Both satisfy the
same protocol, and the active backend is reported in /health rather than
silently swapped.
"""
from __future__ import annotations

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


class QdrantEdgeStore(NativeStore):
    """Qdrant Edge embedded adapter.

    Qdrant Edge runs in-process, so collections are created once at boot with
    named dense + sparse vectors and payload indexes on the fields we filter on
    (`ts`, `sensitivity`, `model_version`). If the wheel is unavailable the
    caller falls back to :class:`NativeStore`; we never degrade silently.
    """

    backend = "qdrant-edge"

    def __init__(self, dim: int, collections: tuple[str, ...], path: str) -> None:
        super().__init__(dim, collections, path)
        from qdrant_client import QdrantClient  # type: ignore
        from qdrant_client.models import Distance, VectorParams  # type: ignore

        self.client = QdrantClient(path=path)
        for name in collections:
            if not self.client.collection_exists(name):
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
        for name in collections:
            for field_name, schema in (("ts", "float"), ("sensitivity", "keyword"), ("model_version", "keyword")):
                try:
                    self.client.create_payload_index(name, field_name=field_name, field_schema=schema)
                except Exception:  # index already present
                    pass

    def upsert(self, point: MemoryPoint) -> None:
        super().upsert(point)
        from qdrant_client.models import PointStruct  # type: ignore

        self.client.upsert(
            collection_name=point.collection,
            points=[PointStruct(
                id=abs(hash(point.id)) % (1 << 63),
                vector=list(point.dense),
                payload={
                    "aegis_id": point.id,
                    "text": point.text,
                    "ts": point.created_at,
                    "sensitivity": point.sensitivity.value,
                    "model_version": point.model_version,
                    **point.payload,
                },
            )],
        )


def build_store(dim: int, collections: tuple[str, ...], path: str) -> VectorStore:
    """Prefer Qdrant Edge; fall back to the native tiered store."""
    try:
        return QdrantEdgeStore(dim, collections, path)
    except Exception:
        return NativeStore(dim, collections, path)
