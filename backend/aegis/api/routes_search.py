"""Retrieval: hybrid search and the agentic answer path."""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException

from ..core.slo import Level
from ..core.tenancy import QuotaExceeded, Scope
from ..node import EdgeNode
from .deps import get_node
from .models import AskRequest, SearchRequest
from .security import Principal, requires

router = APIRouter(prefix="/api/v1", tags=["retrieval"])


@router.post("/search")
async def search(body: SearchRequest, node: EdgeNode = Depends(get_node),
                 who: Principal = Depends(requires(Scope.READ))) -> dict:
    try:
        node.tenants.check_read(who.tenant_id)
    except QuotaExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc

    started = time.perf_counter()
    ok = True
    try:
        result = await node.pipeline.search(
            body.query, k=body.k, collection=body.collection, mode=body.mode,
            explain=body.explain, allow_escalation=body.allow_escalation,
            filters=body.filters, understand=body.understand,
            tenant_id=None if who.anonymous else who.tenant_id,
        )
        return result.as_dict()
    except Exception:
        ok = False
        raise
    finally:
        # Only failures are reported here. A query that completed was already
        # reported by the pipeline, which sees every caller rather than only
        # the ones arriving over HTTP; observing it again would count it twice
        # and halve the apparent breach rate. A query that raised never
        # reached that point, and an SLO computed only from successes is not
        # an SLO — so the transport still owns the failures.
        if not ok:
            node.slo.observe((time.perf_counter() - started) * 1000, False)


@router.post("/ask")
async def ask(body: AskRequest, node: EdgeNode = Depends(get_node),
              who: Principal = Depends(requires(Scope.READ))) -> dict:
    started = time.perf_counter()
    ok = True
    try:
        answer = await node.agent.answer(body.query)
        return answer.as_dict()
    except Exception:
        ok = False
        raise
    finally:
        if not ok:                       # successes are reported by the pipeline
            node.slo.observe((time.perf_counter() - started) * 1000, False)
