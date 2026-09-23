"""Memory: ingest, inspect, delete, compact, consolidate."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..core.tenancy import QuotaExceeded, Scope
from ..node import EdgeNode
from .deps import get_node
from .models import IngestRequest, IngestResponse
from .security import Principal, requires

router = APIRouter(prefix="/api/v1/memory", tags=["memory"])


@router.get("/stats")
async def stats(node: EdgeNode = Depends(get_node)) -> dict:
    return node.store.stats()


@router.post("/ingest", response_model=IngestResponse)
async def ingest(body: IngestRequest, node: EdgeNode = Depends(get_node),
                 who: Principal = Depends(requires(Scope.WRITE))) -> IngestResponse:
    try:
        point = await node.remember(body.text, body.collection, body.payload, body.source,
                                    tenant_id=who.tenant_id)
    except QuotaExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    return IngestResponse(
        id=point.id, collection=point.collection, sensitivity=point.sensitivity.value,
        sync_class=point.sync_class.value, policy_rule=point.payload.get("policy_rule", "default"),
        tier=point.tier.value,
    )


@router.get("/points")
async def points(collection: str | None = None, limit: int = 50, include_superseded: bool = False,
                 node: EdgeNode = Depends(get_node),
                 who: Principal = Depends(requires(Scope.READ))) -> dict:
    rows = [
        p for p in node.store.points.values()
        if (collection is None or p.collection == collection)
        and (include_superseded or p.superseded_by is None)
        and (who.anonymous or p.tenant_id == who.tenant_id)
    ]
    rows.sort(key=lambda p: -p.created_at)
    return {"total": len(rows), "points": [p.summary() for p in rows[:limit]]}


@router.get("/points/{point_id}")
async def point_detail(point_id: str, node: EdgeNode = Depends(get_node),
                       who: Principal = Depends(requires(Scope.READ))) -> dict:
    point = node.store.get(point_id, None if who.anonymous else who.tenant_id)
    if point is None:
        raise HTTPException(status_code=404, detail="no such point")
    detail = point.summary()
    detail["resolved_text"] = node.vault.resolve(point.text)     # local-only de-redaction
    detail["graph"] = {
        "entities": node.graph.by_point.get(point_id, [])[:10],
        "facts": [node.graph.facts[f].as_dict() for f in node.graph.by_point.get(point_id, [])
                  if f in node.graph.facts][:10],
    }
    detail["lineage"] = {
        "derived_from": point.derived_from,
        "superseded_by": point.superseded_by,
        "supersedes": [p.id for p in node.store.points.values() if p.superseded_by == point_id],
    }
    return detail


@router.delete("/points/{point_id}")
async def delete_point(point_id: str, node: EdgeNode = Depends(get_node),
                       who: Principal = Depends(requires(Scope.WRITE))) -> dict:
    from ..sync.crdt import OpKind

    point = node.store.get(point_id, None if who.anonymous else who.tenant_id)
    if point is None:
        raise HTTPException(status_code=404, detail="no such point")
    node.sync.record_local(point, OpKind.DELETE)     # tombstone propagates to the fleet
    node.store.delete(point_id)
    node.pipeline.cache.invalidate()
    return {"deleted": point_id}


@router.post("/compact")
async def compact(node: EdgeNode = Depends(get_node)) -> dict:
    return node.store.compact().as_dict()


@router.post("/consolidate")
async def consolidate(collection: str = "episodic", node: EdgeNode = Depends(get_node)) -> dict:
    report = await node.consolidator.run(collection)
    node.pipeline.cache.invalidate()
    return report.as_dict()
