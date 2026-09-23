"""Node accessor for the routers."""
from __future__ import annotations

from fastapi import HTTPException, Request

from ..node import EdgeNode


def get_node(request: Request) -> EdgeNode:
    node: EdgeNode | None = getattr(request.app.state, "node", None)
    if node is None:
        raise HTTPException(status_code=503, detail="node not initialised")
    return node
