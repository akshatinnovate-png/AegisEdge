"""Index internals: strategies, calibration, planner statistics, traces."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from ..core.tracing import TRACER
from ..node import EdgeNode
from .deps import get_node

router = APIRouter(prefix="/api/v1", tags=["index"])


@router.get("/index")
async def index_report(node: EdgeNode = Depends(get_node)) -> dict:
    store = node.store.store
    if hasattr(store, "index_report"):
        return store.index_report()
    return {"backend": store.backend, "detail": "external backend owns its indexes"}


@router.get("/scheduler")
async def scheduler(node: EdgeNode = Depends(get_node)) -> dict:
    return node.scheduler.snapshot()


@router.get("/traces")
async def traces(limit: int = 3) -> dict:
    return {"summary": TRACER.snapshot(), "recent": TRACER.recent(limit)}
