"""Energy and provenance: the two numbers nobody else publishes.

Both exist to be checked rather than believed. The energy figures say which
source produced them and refuse to dress a model up as a measurement; the
provenance receipt lets anyone with a clone confirm that the process answering
them is the code they are reading.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.tenancy import Scope
from ..node import EdgeNode
from .deps import get_node
from .security import Principal, requires

router = APIRouter(prefix="/api/v1", tags=["receipts"])


class ReceiptBody(BaseModel):
    """A receipt produced elsewhere, to compare against this process."""

    root: str | None = None
    files: dict[str, str] = Field(default_factory=dict)


@router.get("/energy")
async def energy(node: EdgeNode = Depends(get_node),
                 _: Principal = Depends(requires(Scope.READ))) -> dict:
    """Joules per operation, and how many fit in a percent of the battery.

    Latency is the figure everything reports and the wrong one to optimise a
    battery-powered device against: answering in 4 ms while holding four cores
    flat is worse, on a robot, than 12 ms on one core. This is the other half.
    """
    snapshot = node.energy.snapshot()
    headline = None
    query = snapshot["by_kind"].get("query")
    if query and query.get("per_battery_percent"):
        headline = (f"{query['per_battery_percent']:,} answers per 1% of battery "
                    f"at {query['millijoules_per_op']} mJ each")
    elif query:
        headline = (f"{query['millijoules_per_op']} mJ per answer "
                    f"({query['cpu_ms_per_op']} ms of CPU)")
    return {**snapshot, "headline": headline,
            "caveat": ("readings marked `modelled` are CPU-seconds times an assumed "
                       "per-core draw, not a measurement. Where the platform exposes "
                       "RAPL or a battery, that is used instead and says so.")}


@router.get("/provenance")
async def provenance(full: bool = False, node: EdgeNode = Depends(get_node),
                     _: Principal = Depends(requires(Scope.READ))) -> dict:
    """A Merkle root over this process's own source, weights and graphs.

    Clone the repository, run `python3 -m aegis.core.provenance`, and compare
    `root`. If they match, the process answering you is the code that is
    public. `full=true` returns every leaf so a mismatch can be located.
    """
    return node.provenance.full() if full else node.provenance.receipt()


@router.post("/provenance/verify")
async def verify(body: ReceiptBody, node: EdgeNode = Depends(get_node),
                 _: Principal = Depends(requires(Scope.READ))) -> dict:
    """Compare a receipt taken elsewhere against this process, file by file."""
    if not body.root and not body.files:
        raise HTTPException(status_code=400,
                            detail="send at least a root, ideally the files too")
    return node.provenance.verify_against({"root": body.root, "files": body.files})
