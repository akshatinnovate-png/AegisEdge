"""Peer-to-peer mesh: membership and anti-entropy rounds."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from pydantic import BaseModel, Field

from ..node import EdgeNode
from .deps import get_node
from .models import PeerRequest


class ExchangeRequest(BaseModel):
    """One mesh RPC, arriving from a peer device."""

    sender: str
    method: str
    payload: dict = Field(default_factory=dict)

router = APIRouter(prefix="/api/v1/mesh", tags=["mesh"])


@router.get("/status")
async def status(node: EdgeNode = Depends(get_node)) -> dict:
    return node.mesh.snapshot()


@router.post("/peers")
async def add_peer(body: PeerRequest, node: EdgeNode = Depends(get_node)) -> dict:
    peer = node.mesh.add_peer(body.node_id, body.endpoint)
    return {"added": peer.as_dict(), "peers": len(node.mesh.peers)}


@router.delete("/peers/{node_id}")
async def drop_peer(node_id: str, node: EdgeNode = Depends(get_node)) -> dict:
    if node_id not in node.mesh.peers:
        raise HTTPException(status_code=404, detail="no such peer")
    node.mesh.peers.pop(node_id)
    return {"removed": node_id, "peers": len(node.mesh.peers)}


@router.post("/exchange")
async def exchange(body: ExchangeRequest, node: EdgeNode = Depends(get_node)) -> dict:
    """The receiving half of the mesh, when the peer is another device.

    This is deliberately thin. It hands the call straight to the same
    `GossipAgent` an in-process peer would reach, so both sides run one
    implementation of anti-entropy, IBLT reconciliation, causal delivery and
    policy filtering. A second code path for "remote" peers would be a second
    thing to keep correct, and the first one to drift.
    """
    try:
        return await node.mesh.handle(body.sender, body.method, body.payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/offline")
async def offline(on: bool = True, node: EdgeNode = Depends(get_node)) -> dict:
    """Pull this device's radio, or put it back.

    The mesh is device-to-device, so being offline here is not the same as
    losing the cloud uplink: the node keeps serving locally and keeps queueing
    its own operations, it simply cannot reach its peers.
    """
    link = node.mesh_link
    setter = getattr(link, "set_offline", None)
    if not callable(setter):
        raise HTTPException(status_code=409,
                            detail="this node's mesh link has no radio to pull")
    state = setter(on)
    node.bus.publish("mesh", "radio", offline=state,
                     message=("mesh radio <b>down</b> — peers unreachable, local memory "
                              "still authoritative" if state else
                              "mesh radio <b>up</b> — peers reachable again"))
    return {"offline": state, "link": link.snapshot()}


@router.post("/round")
async def round_now(peers: int = 2, node: EdgeNode = Depends(get_node)) -> dict:
    if not node.mesh.peers:
        raise HTTPException(status_code=409, detail="no peers known to this node")
    return {"rounds": await node.mesh.round(peers), "mesh": node.mesh.snapshot()}
