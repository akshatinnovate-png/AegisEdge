"""Retrieval: hybrid search and the agentic answer path."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from ..node import EdgeNode
from .deps import get_node
from .models import AskRequest, SearchRequest

router = APIRouter(prefix="/api/v1", tags=["retrieval"])


@router.post("/search")
async def search(body: SearchRequest, node: EdgeNode = Depends(get_node)) -> dict:
    result = await node.pipeline.search(
        body.query, k=body.k, collection=body.collection, mode=body.mode,
        explain=body.explain, allow_escalation=body.allow_escalation, filters=body.filters,
    )
    return result.as_dict()


@router.post("/ask")
async def ask(body: AskRequest, node: EdgeNode = Depends(get_node)) -> dict:
    answer = await node.agent.answer(body.query)
    return answer.as_dict()
