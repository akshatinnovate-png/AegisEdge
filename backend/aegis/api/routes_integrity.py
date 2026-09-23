"""Durability: archives, fsck, scrub, repair and point-in-time restore."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..core.tenancy import Scope
from ..node import EdgeNode
from .deps import get_node
from .models import RestoreRequest
from .security import Principal, requires

router = APIRouter(prefix="/api/v1/integrity", tags=["integrity"])


@router.get("/status")
async def status(node: EdgeNode = Depends(get_node),
                 _: Principal = Depends(requires(Scope.READ))) -> dict:
    return {"segments": node.segments.snapshot(), "wal": node.store.wal.stats(),
            "repair": node.repair.snapshot(), "audit": node.audit.snapshot()}


@router.post("/archive")
async def archive(node: EdgeNode = Depends(get_node),
                  _: Principal = Depends(requires(Scope.ADMIN))) -> dict:
    sealed = node.store.archive()
    return {"sealed": sealed, "segments": node.segments.snapshot()}


@router.post("/fsck")
async def fsck(repair: bool = False, node: EdgeNode = Depends(get_node),
               _: Principal = Depends(requires(Scope.ADMIN))) -> dict:
    report = node.segments.fsck(repair=repair)
    return {**report.as_dict(), "records_at_risk": node.segments.lost_records(report)}


@router.post("/scrub")
async def scrub(repair: bool = True, node: EdgeNode = Depends(get_node),
                _: Principal = Depends(requires(Scope.ADMIN))) -> dict:
    """Read everything, find silent corruption, heal it from peers or cloud."""
    return await node.repair.scrub(repair=repair)


@router.get("/generations")
async def generations(node: EdgeNode = Depends(get_node),
                      _: Principal = Depends(requires(Scope.READ))) -> dict:
    return {"generations": node.segments.generations(),
            "current": node.segments.manifest.generation}


@router.post("/restore")
async def restore(body: RestoreRequest, node: EdgeNode = Depends(get_node),
                  _: Principal = Depends(requires(Scope.ADMIN))) -> dict:
    available = {row["generation"] for row in node.segments.generations()}
    if body.generation not in available and body.generation != 0:
        raise HTTPException(status_code=404, detail=f"no generation {body.generation}")
    dropped = node.segments.restore_to(body.generation)
    node.audit.record("pitr_restore", str(body.generation), dropped=dropped)
    return {"restored_to": body.generation, "segments_dropped": dropped,
            "note": "restart the node to rebuild in-memory state from the restored segments"}
