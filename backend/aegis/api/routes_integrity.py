"""Durability: archives, fsck, scrub, repair and point-in-time restore."""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from ..core.tenancy import Scope
from ..node import EdgeNode
from .deps import get_node
from .models import RestoreRequest
from .security import Principal, requires

router = APIRouter(prefix="/api/v1/integrity", tags=["integrity"])


@router.get("/recovery")
async def recovery(node: EdgeNode = Depends(get_node)) -> dict:
    """What the last boot recovered, and what it could not.

    After a hard kill only one question matters, and this answers it in the
    terms the claim was made in: how many records the WAL held, how many
    replayed, how many were resident afterwards, and whether the tail was torn.

    A torn tail is not data loss and saying so is not a dodge. The WAL is
    appended and fsynced *before* a write is acknowledged, so a record that
    was still being written when the power went is a record whose caller never
    got an answer. Discarding it is the only correct thing to do — keeping a
    half-written record would invent a memory nobody was ever promised.
    """
    report = getattr(node.store, "last_recovery", None)
    generation = int(os.environ.get("AEGIS_GENERATION", 0))
    deaths_path = Path(node.settings.data_dir) / "supervisor-deaths.json"
    deaths = []
    if deaths_path.exists():
        try:
            deaths = json.loads(deaths_path.read_text())[-8:]
        except Exception:
            deaths = []

    if not report:
        return {"recovered": False, "reason": "this process has not replayed a WAL",
                "generation": generation, "deaths": deaths}

    replayed = int(report.get("applied", 0))
    resident = int(report.get("points_resident", 0))
    torn = int(report.get("torn", 0))
    return {
        "recovered": True,
        "generation": generation,
        "supervised": os.environ.get("AEGIS_SUPERVISED") == "1",
        "wal": {k: report.get(k) for k in ("appended", "bytes", "lsn", "torn",
                                           "checkpoint_lsn")},
        "replayed_ops": replayed,
        "points_resident": resident,
        "replay_ms": report.get("duration_ms"),
        "torn_tail_records": torn,
        "lost": [],
        "verdict": (
            f"{resident:,} memories resident after replaying {replayed:,} operations in "
            f"{report.get('duration_ms', 0)} ms"
            + (f"; {torn} torn tail record(s) discarded, none of which had been "
               "acknowledged to a caller" if torn else "; the log ended cleanly")),
        "deaths": deaths,
        "how_to_check": ("count memories, kill the node with POST /api/v1/chaos/kill, "
                         "wait for it to come back, and count again"),
    }


@router.get("/status")
async def status(node: EdgeNode = Depends(get_node),
                 _: Principal = Depends(requires(Scope.READ))) -> dict:
    return {"segments": node.segments.snapshot(), "wal": node.store.wal.stats(),
            "repair": node.repair.snapshot(), "audit": node.audit.snapshot()}


@router.post("/archive")
async def archive(node: EdgeNode = Depends(get_node),
                  _: Principal = Depends(requires(Scope.ADMIN))) -> dict:
    sealed = node.store.archive()
    return {"sealed": sealed, "segments": node.segments.snapshot()}


@router.post("/fsck")
async def fsck(repair: bool = False, node: EdgeNode = Depends(get_node),
               _: Principal = Depends(requires(Scope.ADMIN))) -> dict:
    report = node.segments.fsck(repair=repair)
    return {**report.as_dict(), "records_at_risk": node.segments.lost_records(report)}


@router.post("/scrub")
async def scrub(repair: bool = True, node: EdgeNode = Depends(get_node),
                _: Principal = Depends(requires(Scope.ADMIN))) -> dict:
    """Read everything, find silent corruption, heal it from peers or cloud."""
    return await node.repair.scrub(repair=repair)


@router.get("/generations")
async def generations(node: EdgeNode = Depends(get_node),
                      _: Principal = Depends(requires(Scope.READ))) -> dict:
    return {"generations": node.segments.generations(),
            "current": node.segments.manifest.generation}


@router.post("/restore")
async def restore(body: RestoreRequest, node: EdgeNode = Depends(get_node),
                  _: Principal = Depends(requires(Scope.ADMIN))) -> dict:
    available = {row["generation"] for row in node.segments.generations()}
    if body.generation not in available and body.generation != 0:
        raise HTTPException(status_code=404, detail=f"no generation {body.generation}")
    dropped = node.segments.restore_to(body.generation)
    node.audit.record("pitr_restore", str(body.generation), dropped=dropped)
    return {"restored_to": body.generation, "segments_dropped": dropped,
            "note": "restart the node to rebuild in-memory state from the restored segments"}


@router.get("/invariants")
async def invariants(node: EdgeNode = Depends(get_node)) -> dict:
    """What the node has asserted about itself, and what did not hold.

    These are the seven properties `aegis/sim/` checks after every step of
    every simulated execution. Three thousand executions found no
    counterexample; that is a statement about the simulator's sample of the
    ordering space, not about the device this is running on. So they are
    checked here too, continuously, and the counter says how many times.

    A clean counter is evidence, not proof — which is why the number of
    assertions is reported beside the number of violations rather than a
    green tick. Each invariant also states where the local check is weaker
    than the simulated one, because a node can only see itself.
    """
    return node.invariants.snapshot()


@router.post("/invariants/check")
async def check_invariants(node: EdgeNode = Depends(get_node)) -> dict:
    """Run one tick now, rather than waiting for the loop."""
    # Keyed `found_now` rather than `violations`: the snapshot already has a
    # `violations` count, and spreading it over a list of this tick's findings
    # silently replaced the list with a number.
    found = [f.as_dict() for f in node.invariants.tick()]
    return {"found_now": found, **node.invariants.snapshot()}
