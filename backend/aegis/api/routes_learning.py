"""Learning: feedback, the on-device adapter, and federated rounds."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..learning.adapter import RetrievalAdapter
from ..learning.federated import FederatedClient
from ..learning.privacy import PrivacyBudget
from ..node import EdgeNode
from .deps import get_node
from .models import FeedbackRequest, FederatedRoundRequest

router = APIRouter(prefix="/api/v1/learning", tags=["learning"])


@router.get("/status")
async def status(node: EdgeNode = Depends(get_node)) -> dict:
    return {
        "adapter": node.adapter.snapshot(),
        "pending_feedback": len(node.feedback_buffer),
        "federation": node.federation.snapshot(),
        "coordinator": node.coordinator.snapshot(),
    }


@router.post("/feedback")
async def feedback(body: FeedbackRequest, node: EdgeNode = Depends(get_node)) -> dict:
    result = await node.feedback(body.query, body.chosen_id, body.rejected_id, body.weight)
    if not result.get("accepted"):
        raise HTTPException(status_code=404, detail=result.get("reason", "rejected"))
    return result


@router.post("/round")
async def federated_round(body: FederatedRoundRequest | None = None,
                          node: EdgeNode = Depends(get_node)) -> dict:
    """Run one secure-aggregation round.

    Simulated cohort by default: one real device cannot demonstrate that the
    masks cancel, and the report states the cohort size the configured epsilon
    would actually need.
    """
    body = body or FederatedRoundRequest()
    cohort = body.cohort or [node.settings.node_id] + [f"peer-{i}" for i in range(body.simulate_peers)]
    round_id = node.coordinator.round_id + 1

    contribution = node.federation.contribute(cohort, round_id)
    if contribution is None:
        raise HTTPException(status_code=409, detail="privacy budget exhausted for this node")
    contributions = [contribution]

    # Simulated peers get their own adapters and their own privacy budgets.
    # Re-using this node's budget for them would both corrupt the accounting
    # and drain the real device's allowance to fake a cohort.
    for peer in cohort[1:]:
        simulated = FederatedClient(
            peer, RetrievalAdapter(node.adapter.dim, rank=node.adapter.rank,
                                   alpha=node.adapter.alpha),
            PrivacyBudget(epsilon_total=node.settings.learning.epsilon_total),
            epsilon_per_round=node.settings.learning.epsilon_per_round,
            differential_privacy=node.settings.learning.differential_privacy,
        )
        peer_contribution = simulated.contribute(cohort, round_id)
        if peer_contribution is not None:
            contributions.append(peer_contribution)

    result = node.coordinator.aggregate(cohort, contributions)
    result["simulated_peers"] = max(0, len(contributions) - 1)
    return result


@router.post("/adopt")
async def adopt(node: EdgeNode = Depends(get_node)) -> dict:
    if node.coordinator.global_parameters is None:
        raise HTTPException(status_code=409, detail="no aggregated parameters yet")
    node.federation.adopt(node.coordinator.global_parameters)
    return {"adopted": True, "adapter": node.adapter.snapshot()}
