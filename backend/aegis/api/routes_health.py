"""Health, node state, metrics, audit and policy introspection."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse

from ..core.metrics import METRICS
from ..node import EdgeNode
from .deps import get_node

router = APIRouter(prefix="/api/v1", tags=["node"])


@router.get("/health")
async def health(node: EdgeNode = Depends(get_node)) -> dict:
    return node.health()


@router.get("/node/state")
async def node_state(node: EdgeNode = Depends(get_node)) -> dict:
    return node.state()


@router.get("/node/config")
async def node_config(node: EdgeNode = Depends(get_node)) -> dict:
    return node.settings.as_dict()

@router.get("/node/subsystems")
async def subsystems(node: EdgeNode = Depends(get_node)) -> dict:
    return {
        "supervisor": node.supervisor.health(),
        "inference": {
            "embedder": node.embedder.snapshot(),
            "reranker": node.reranker.snapshot(),
            "sparse": node.sparse.snapshot(),
            "classifier": node.classifier.snapshot(),
            "governor": node.governor.snapshot(),
            "triton": node.triton.snapshot(),
            "registry": node.registry.snapshot(),
        },
        "retrieval": node.pipeline.snapshot(),
        "policy": node.policy.snapshot(),
        "audit": node.audit.snapshot(),
        "vault": node.vault.snapshot(),
        "bus": {"published": node.bus.published, "subscribers": node.bus.subscribers},
    }


@router.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    return METRICS.prometheus()


@router.get("/metrics/json")
async def metrics_json() -> dict:
    return METRICS.snapshot()


@router.get("/audit")
async def audit(limit: int = 50, node: EdgeNode = Depends(get_node)) -> dict:
    return {"chain": node.audit.snapshot(), "entries": node.audit.tail(limit)}


@router.get("/policy")
async def policy(node: EdgeNode = Depends(get_node)) -> dict:
    return {"policy": node.policy.document, "stats": node.policy.snapshot()}


@router.post("/policy/reload")
async def policy_reload(node: EdgeNode = Depends(get_node)) -> dict:
    changed = node.policy.reload()
    node.bus.publish("alerts", "policy_reloaded", level="ok" if changed else "info",
                     changed=changed, message="policy reloaded" if changed else "policy unchanged")
    return {"reloaded": changed, "stats": node.policy.snapshot()}
