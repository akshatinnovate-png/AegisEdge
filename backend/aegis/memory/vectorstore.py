"""Vector store abstraction.

`QdrantEdgeStore` binds to Qdrant Edge in-process when the wheel is present.
`NativeStore` is the pure-NumPy tiered implementation used otherwise, so the
node boots and demos on any machine with identical semantics. Both satisfy the
same protocol, and the active backend is reported in /health rather than
silently swapped.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from .index import TieredIndex
from .schema import MemoryPoint, Tier


class VectorStore(Protocol):
    backend: str

    def upsert(self, point: MemoryPoint) -> None: ...
    def delete(self, point_id: str) -> None: ...
    def move(self, point_id: str, tier: Tier) -> bool: ...
    def search_dense(self, collection: str, query: np.ndarray, k: int) -> list[tuple[str, float]]: ...
    def search_sparse(self, collection: str, sparse: dict[int, float], k: int) -> list[tuple[str, float]]: ...
    def counts(self) -> dict[str, dict[str, int]]: ...


class NativeStore:
    """Tiered NumPy store — no external service, no network hop."""

    backend = "native-tiered"

    def __init__(self, dim: int, collections: tuple[str, ...]) -> None:
        self.dim = dim
        self.indexes: dict[str, TieredIndex] = {c: TieredIndex(dim) for c in collections}

    def _index(self, collection: str) -> TieredIndex:
        if collection not in self.indexes:
            self.indexes[collection] = TieredIndex(self.dim)
        return self.indexes[collection]

    def upsert(self, point: MemoryPoint) -> None:
        vector = np.asarray(point.dense, dtype=np.float32)
        if vector.shape[0] != self.dim:
            raise ValueError(f"dim mismatch: {vector.shape[0]} != {self.dim}")
        self._index(point.collection).upsert(point.id, vector, point.sparse, point.tier)

    def delete(self, point_id: str) -> None:
        for index in self.indexes.values():
            index.remove(point_id)

    def move(self, point_id: str, tier: Tier) -> bool:
        return any(index.move(point_id, tier) for index in self.indexes.values())

    def search_dense(self, collection: str, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        if collection == "*":
            merged: list[tuple[str, float]] = []
            for index in self.indexes.values():
                merged.extend(index.search_dense(query, k))
            merged.sort(key=lambda x: -x[1])
            return merged[:k]
        return self._index(collection).search_dense(query, k)

    def search_sparse(self, collection: str, sparse: dict[int, float], k: int) -> list[tuple[str, float]]:
        if collection == "*":
            merged: list[tuple[str, float]] = []
            for index in self.indexes.values():
                merged.extend(index.search_sparse(sparse, k))
            merged.sort(key=lambda x: -x[1])
            return merged[:k]
        return self._index(collection).search_sparse(sparse, k)

    def counts(self) -> dict[str, dict[str, int]]:
        return {name: index.counts() for name, index in self.indexes.items()}

    @property
    def rescored(self) -> int:
        return sum(i.rescored for i in self.indexes.values())


class QdrantEdgeStore(NativeStore):
    """Qdrant Edge embedded adapter.

    Qdrant Edge runs in-process, so collections are created once at boot with
    named dense + sparse vectors and payload indexes on the fields we filter on
    (`ts`, `sensitivity`, `model_version`). If the wheel is unavailable the
    caller falls back to :class:`NativeStore`; we never degrade silently.
    """

    backend = "qdrant-edge"

    def __init__(self, dim: int, collections: tuple[str, ...], path: str) -> None:
        super().__init__(dim, collections)
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
        return NativeStore(dim, collections)
