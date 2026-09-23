"""SLOs and the degradation ladder."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..core.slo import FEATURES, RUNG_REASON, Level
from ..core.tenancy import Scope
from ..node import EdgeNode
from .deps import get_node
from .models import DegradationRequest
from .security import Principal, requires

router = APIRouter(prefix="/api/v1/slo", tags=["slo"])


@router.get("")
async def status(node: EdgeNode = Depends(get_node),
                 _: Principal = Depends(requires(Scope.READ))) -> dict:
    return {
        **node.slo.snapshot(),
        "ladder": {level.name: RUNG_REASON[level] for level in Level},
        "features": {name: rung.name for name, rung in FEATURES.items()},
    }


@router.post("/override")
async def override(body: DegradationRequest, node: EdgeNode = Depends(get_node),
                   _: Principal = Depends(requires(Scope.ADMIN))) -> dict:
    """Pin the ladder, or release it back to automatic control."""
    if body.level is None:
        node.slo.override(None)
        return {"override": None, "level": node.slo.level.name}
    try:
        target = Level[body.level.upper()]
    except KeyError as exc:
        raise HTTPException(status_code=400,
                            detail=f"unknown level; expected one of "
                                   f"{[l.name for l in Level]}") from exc
    node.slo.override(target)
    node.audit.record("degradation_override", target.name)
    return {"override": target.name, "disabled": node.slo.disabled(),
            "reason": RUNG_REASON[target]}
