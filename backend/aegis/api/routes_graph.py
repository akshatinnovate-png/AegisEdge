"""Knowledge graph: entities, relations, paths and time travel."""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException

from ..core.tenancy import Scope
from ..node import EdgeNode
from .deps import get_node
from .models import PathRequest
from .security import Principal, requires

router = APIRouter(prefix="/api/v1/graph", tags=["graph"])


@router.get("/stats")
async def stats(node: EdgeNode = Depends(get_node),
                _: Principal = Depends(requires(Scope.READ))) -> dict:
    return node.graph.snapshot()


@router.get("/entities")
async def entities(type: str | None = None, limit: int = 50,
                   node: EdgeNode = Depends(get_node),
                   _: Principal = Depends(requires(Scope.READ))) -> dict:
    rows = [e for e in node.graph.entities.values() if type is None or e.type.value == type]
    rows.sort(key=lambda e: -len(e.mentions))
    return {"total": len(rows), "entities": [e.as_dict() for e in rows[:limit]]}


@router.get("/entities/{entity_id}")
async def entity_detail(entity_id: str, as_of: float | None = None,
                        node: EdgeNode = Depends(get_node),
                        _: Principal = Depends(requires(Scope.READ))) -> dict:
    entity = node.graph.entities.get(entity_id)
    if entity is None:
        raise HTTPException(status_code=404, detail="no such entity")
    facts = node.graph.neighbours(entity_id, as_of=as_of)
    return {"entity": entity.as_dict(),
            "facts": [f.as_dict() for f in facts],
            "memories": entity.mentions[:20]}


@router.post("/paths")
async def paths(body: PathRequest, node: EdgeNode = Depends(get_node),
                _: Principal = Depends(requires(Scope.READ))) -> dict:
    found = node.graph.paths(body.source, body.target, body.max_hops,
                             valid_time=body.valid_time, as_of=body.as_of)
    return {"source": body.source, "target": body.target,
            "paths": [p.as_dict() for p in found]}


@router.get("/as-of")
async def as_of(when: float | None = None, valid_time: float | None = None,
                node: EdgeNode = Depends(get_node),
                _: Principal = Depends(requires(Scope.READ))) -> dict:
    """What this node believed at a moment — the question incident reviews ask."""
    return node.graph.as_of(when or time.time(), valid_time)


@router.get("/diff")
async def diff(earlier: float, later: float | None = None,
               node: EdgeNode = Depends(get_node),
               _: Principal = Depends(requires(Scope.READ))) -> dict:
    return node.graph.diff_beliefs(earlier, later or time.time())


@router.post("/facts/{fact_id}/retract")
async def retract(fact_id: str, node: EdgeNode = Depends(get_node),
                  _: Principal = Depends(requires(Scope.WRITE))) -> dict:
    if not node.graph.retract(fact_id):
        raise HTTPException(status_code=404, detail="no such live fact")
    node.audit.record("fact_retracted", fact_id)
    return {"retracted": fact_id, "note": "belief withdrawn; the fact remains auditable"}
