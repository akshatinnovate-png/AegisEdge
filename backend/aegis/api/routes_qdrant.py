"""Qdrant: the schema it holds, the plan it ran, and a race against the interpreter.

The inspection interface is a requirement, not a flourish, and the thing most
worth inspecting about a vector database is the query it was actually asked —
not a diagram of the query someone intended. Every field here is read from the
live client or from the last plan the engine executed.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ..core.tenancy import Scope
import numpy as np

from ..memory.filters import Filter
from ..retrieval.projection import project, scale
from ..node import EdgeNode
from .deps import get_node
from .models import SearchRequest
from .security import Principal, requires

router = APIRouter(prefix="/api/v1", tags=["qdrant"])


def _store(node: EdgeNode):
    store = node.store.store
    if not hasattr(store, "hybrid"):
        raise HTTPException(status_code=503,
                            detail="this node is running the internal store, not Qdrant")
    return store


@router.get("/qdrant")
async def qdrant(node: EdgeNode = Depends(get_node),
                 who: Principal = Depends(requires(Scope.READ))) -> dict:
    """What the engine is, what it holds, and which path queries take."""
    store = _store(node)
    pipeline = node.pipeline
    counts: dict[str, Any] = {}
    for name in store.collections:
        try:
            info = store.client.get_collection(name)
            counts[name] = {"points": getattr(info, "points_count", None),
                            "schema": store.hybrid.schema.get(name)}
        except Exception as exc:                 # a collection can be mid-creation
            counts[name] = {"error": type(exc).__name__, "schema": store.hybrid.schema.get(name)}
    indexes = store.hybrid.payload_indexes
    return {
        "backend": store.backend,
        "url": store.url,
        "hybrid": store.hybrid.snapshot(),
        "collections": counts,
        "payload_indexes": {
            "requested": len(indexes),
            # Local mode says outright that payload indexes do nothing there;
            # reporting them as live would be the one lie this panel could tell.
            "live": sum(1 for v in indexes.values() if v not in
                        ("ignored-by-local-mode",) and not v.startswith("skipped")),
            "ignored_by_local_mode": sum(1 for v in indexes.values()
                                         if v == "ignored-by-local-mode"),
            "fields": sorted({key.split(".", 1)[1] for key in indexes}),
        },
        "query_path": {
            **pipeline.router.snapshot(),
            "engine_queries": pipeline.engine_queries,
            "engine_refusals": pipeline.engine_refusals,
            "interpreter_queries": max(pipeline.queries - pipeline.engine_queries, 0),
        },
        "last_plan": store.last_plan.as_dict() if store.last_plan is not None else None,
        "upserts": store.upserts,
    }


@router.post("/qdrant/bakeoff")
async def bakeoff(body: SearchRequest, node: EdgeNode = Depends(get_node),
                  who: Principal = Depends(requires(Scope.READ))) -> dict:
    """Run one query down both paths and report the difference.

    The same measurement `scripts/qdrant_bakeoff.py` makes over a corpus, for a
    single query, so the console can show it rather than cite it.
    """
    store = _store(node)
    vector = node.embedder.embed_sync([body.query])[0]
    sparse = node.sparse.encode(body.query)
    spec = Filter.parse(body.filters)
    report = store.bake_off(body.collection, vector, sparse, body.k, spec=spec)
    report["query"] = body.query
    return report


@router.get("/qdrant/facets")
async def facets(collection: str = "episodic", key: str = "sensitivity",
                 limit: int = 8, node: EdgeNode = Depends(get_node),
                 who: Principal = Depends(requires(Scope.READ))) -> dict:
    """Payload value counts, computed by the engine rather than by a scan here."""
    store = _store(node)
    if collection not in store.collections:
        raise HTTPException(status_code=404, detail=f"no collection {collection!r}")
    return {"collection": collection, "key": key,
            "hits": store.hybrid.facet(collection, key, limit),
            "computed_by": store.backend}


@router.get("/qdrant/map")
async def vector_map(collection: str = "episodic", limit: int = 600, query: str = "",
                     k: int = 5, node: EdgeNode = Depends(get_node),
                     who: Principal = Depends(requires(Scope.READ))) -> dict:
    """The corpus as the engine holds it, projected to two dimensions.

    Any two-dimensional picture of a 256-dimensional space is a lie of
    compression, so `explained_variance` comes back with it: it is the share of
    variance the two axes actually carry, and without it an operator would read
    adjacency on the screen as similarity in the space.
    """
    store = _store(node)
    if collection not in store.collections:
        raise HTTPException(status_code=404, detail=f"no collection {collection!r}")
    ids, vectors, payloads = store.hybrid.sample(collection, max(1, min(limit, 5_000)))
    if not ids:
        return {"collection": collection, "points": [], "sampled": 0,
                "explained_variance": 0.0, "backend": store.backend,
                "detail": "nothing stored in this collection yet"}

    projection = project(vectors)
    hits: dict[str, int] = {}
    query_point = None
    if query:
        vector = node.embedder.embed_sync([query])[0]
        sparse = node.sparse.encode(query)
        found, plan = store.search_native(collection, vector, sparse, k)
        if plan.fell_back:
            found = store.search_dense(collection, np.asarray(vector, dtype=np.float32), k)
        hits = {point_id: rank for rank, (point_id, _) in enumerate(found, start=1)}
        query_point = projection.project(vector)

    placed = scale(projection.coords, query_point)
    # This plot carries memory text. It is scoped like every other route that
    # returns content: one tenant's console must not be able to read another
    # tenant's memories because the endpoint draws a picture rather than a list.
    points = [
        {"id": point_id, "xy": xy, "rank": hits.get(point_id),
         "sensitivity": payload.get("sensitivity"), "stale": bool(payload.get("stale")),
         "text": (payload.get("text") or "")[:120]}
        for point_id, xy, payload in zip(ids, placed["points"], payloads)
        if who.anonymous or payload.get("tenant_id") == who.tenant_id
    ]
    return {
        "collection": collection, "backend": store.backend,
        "sampled": len(points), "of_sample": len(ids),
        "explained_variance": projection.explained,
        "points": points, "query": placed["query"], "query_text": query,
        "matched": len(hits),
    }
