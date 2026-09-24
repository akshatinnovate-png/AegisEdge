"""Chaos endpoints — prove the resilience claims live."""
from __future__ import annotations

import asyncio
import os
import signal

from fastapi import APIRouter, Depends, HTTPException

from ..node import EdgeNode
from .deps import get_node
from .models import ChaosRequest

router = APIRouter(prefix="/api/v1/chaos", tags=["chaos"])


@router.get("")
async def available(node: EdgeNode = Depends(get_node)) -> dict:
    return node.chaos.snapshot()


@router.post("/kill")
async def kill(sig: int = signal.SIGKILL, delay_ms: int = 350,
               node: EdgeNode = Depends(get_node)) -> dict:
    """Pull the rug out. The hardest test this system has, on demand.

    A durability claim that cannot be tested is a slogan. "Nothing becomes
    searchable before it is recoverable" is either true — in which case losing
    the process mid-write costs milliseconds and no memories — or it is not,
    and the only way to tell from outside is to kill it and look.

    SIGKILL is the default because it is the only interesting one: no handler
    runs, no buffer is flushed, nothing is tidied. SIGTERM would let the node
    shut down politely, which proves nothing about a battery being removed.

    Refused unless a supervisor is there to restart the process. A kill button
    with nothing behind it is not a demonstration, it is the end of the demo.
    """
    if os.environ.get("AEGIS_SUPERVISED") != "1":
        raise HTTPException(
            status_code=409,
            detail=("no supervisor: nothing would restart this process. Run it "
                    "under `python3 scripts/supervise.py` and try again."))
    if sig not in (signal.SIGKILL, signal.SIGTERM, signal.SIGABRT):
        raise HTTPException(status_code=400, detail="signal must be 9, 15 or 6")

    resident = len(node.store.points)
    lsn = node.store.wal.lsn
    node.audit.record("chaos_kill", f"signal={sig} resident={resident} lsn={lsn}")
    node.bus.publish(
        "alerts", "killing", level="warn", signal=sig, resident=resident, wal_lsn=lsn,
        message=(f"receiving <b>SIG{signal.Signals(sig).name[3:]}</b> in {delay_ms} ms · "
                 f"<b>{resident}</b> memories resident · WAL at {lsn}"))

    async def die() -> None:
        # Long enough for this response to reach the browser, and no longer.
        # The point is to die without warning, not to wind down.
        await asyncio.sleep(max(delay_ms, 50) / 1000.0)
        os.kill(os.getpid(), sig)

    asyncio.get_running_loop().create_task(die())
    return {
        "killing": True, "signal": sig, "signal_name": signal.Signals(sig).name,
        "in_ms": delay_ms,
        "resident_before": resident,
        "wal_lsn_before": lsn,
        "generation": int(os.environ.get("AEGIS_GENERATION", 0)),
        "promise": ("every memory acknowledged before this point was written to the "
                    "WAL and fsynced before it was answered, so all of them should "
                    "come back. Check /api/v1/integrity/recovery once it does."),
    }


@router.post("/{fault}")
async def inject(fault: str, body: ChaosRequest | None = None,
                 node: EdgeNode = Depends(get_node)) -> dict:
    body = body or ChaosRequest()
    try:
        return await node.chaos.inject(fault, body.duration_s, **body.params)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
