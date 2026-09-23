"""Chaos endpoints — prove the resilience claims live."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..node import EdgeNode
from .deps import get_node
from .models import ChaosRequest

router = APIRouter(prefix="/api/v1/chaos", tags=["chaos"])


@router.get("")
async def available(node: EdgeNode = Depends(get_node)) -> dict:
    return node.chaos.snapshot()


@router.post("/{fault}")
async def inject(fault: str, body: ChaosRequest | None = None,
                 node: EdgeNode = Depends(get_node)) -> dict:
    body = body or ChaosRequest()
    try:
        return await node.chaos.inject(fault, body.duration_s, **body.params)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
