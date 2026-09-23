"""Peer-to-peer mesh: membership and anti-entropy rounds."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..node import EdgeNode
from .deps import get_node
from .models import PeerRequest

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


@router.post("/round")
async def round_now(peers: int = 2, node: EdgeNode = Depends(get_node)) -> dict:
    if not node.mesh.peers:
        raise HTTPException(status_code=409, detail="no peers known to this node")
    return {"rounds": await node.mesh.round(peers), "mesh": node.mesh.snapshot()}
