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
from ..memory.filters import Filter
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
async def qdrant(node: EdgeNode = Depends(get_node)) -> dict:
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
                 limit: int = 8, node: EdgeNode = Depends(get_node)) -> dict:
    """Payload value counts, computed by the engine rather than by a scan here."""
    store = _store(node)
    if collection not in store.collections:
        raise HTTPException(status_code=404, detail=f"no collection {collection!r}")
    return {"collection": collection, "key": key,
            "hits": store.hybrid.facet(collection, key, limit),
            "computed_by": store.backend}
