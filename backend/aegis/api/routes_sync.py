"""Sync: status, manual reconcile, conflict review, fleet injection."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..node import EdgeNode
from .deps import get_node
from .models import ReviewRequest, SyncTriggerRequest

router = APIRouter(prefix="/api/v1/sync", tags=["sync"])


@router.get("/status")
async def status(node: EdgeNode = Depends(get_node)) -> dict:
    return node.sync.status()


@router.post("/trigger")
async def trigger(body: SyncTriggerRequest | None = None, node: EdgeNode = Depends(get_node)) -> dict:
    result = await node.sync.reconcile(trigger=(body.reason if body else "manual"))
    return {**node.sync.status(), "cycle": result}


@router.get("/conflicts")
async def conflicts(node: EdgeNode = Depends(get_node)) -> dict:
    return {
        "summary": node.sync.arbiter.snapshot(),
        "records": [r.as_dict() for r in node.sync.arbiter.records[-50:]],
        "pending_review": [r.as_dict() for r in node.sync.arbiter.review_queue],
    }


@router.post("/conflicts/review")
async def review(body: ReviewRequest, node: EdgeNode = Depends(get_node)) -> dict:
    if not node.sync.arbiter.review(body.point_id, body.keep):
        raise HTTPException(status_code=404, detail="no pending conflict for that point")
    return {"reviewed": body.point_id, "kept": body.keep,
            "pending": len(node.sync.arbiter.review_queue)}


@router.get("/digest")
async def digest(node: EdgeNode = Depends(get_node)) -> dict:
    local = node.sync.tree.digest()
    return {"local": local.as_dict(), "divergent_buckets": node.sync.divergent,
            "bytes_saved": node.sync.bytes_saved}


@router.post("/fleet/inject")
async def fleet_inject(text: str, collection: str = "semantic",
                       node: EdgeNode = Depends(get_node)) -> dict:
    """Demo hook: a *different* device learned something. Pull must bring it here."""
    if not hasattr(node.transport, "inject"):
        raise HTTPException(status_code=400, detail="transport does not support injection")
    from ..core.ids import short_id

    vector = node.embedder.embed_sync([text])[0].tolist()
    point_id = short_id("fleet")
    op = node.transport.inject(point_id, node.clock.now().pack(), {
        "collection": collection, "text": text, "dense": vector,
        "sensitivity": "internal", "confidence": 0.9, "device_id": "edge-fleet",
    })
    return {"injected": point_id, "op_id": op.op_id, "cloud_ops": len(node.transport.ops)}
