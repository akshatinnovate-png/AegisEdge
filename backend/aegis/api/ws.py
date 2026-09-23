"""Multiplexed WebSocket gateway.

One socket, logical channels, heartbeats. Subscribers get a bounded queue with
drop-oldest backpressure, so a phone on a bad connection degrades to sampled
telemetry instead of stalling the node that is feeding it.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import time

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from ..node import EdgeNode

router = APIRouter(tags=["stream"])

HEARTBEAT_S = 10.0


@router.websocket("/api/v1/stream")
async def stream(websocket: WebSocket, channels: str = Query(default="")) -> None:
    await websocket.accept()
    node: EdgeNode = websocket.app.state.node
    selected = {c.strip() for c in channels.split(",") if c.strip()}
    subscription = node.bus.subscribe(selected or None)

    async def pump() -> None:
        async for event in subscription:
            await websocket.send_text(json.dumps(event.as_dict(), default=str))

    async def heartbeat() -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_S)
            await websocket.send_text(json.dumps({
                "channel": "telemetry", "kind": "heartbeat", "ts": time.time(),
                "dropped": subscription.dropped,
            }))

    try:
        await websocket.send_text(json.dumps({
            "channel": "telemetry", "kind": "hello", "ts": time.time(),
            "node": node.state(), "memory": node.store.stats(), "sync": node.sync.status(),
            "channels": sorted(selected) or list(node.bus.CHANNELS),
            "message": f"stream open · node <b>{node.settings.node_id}</b>",
        }, default=str))
        for event in node.bus.replay(30):                 # backfill so the log is never empty
            await websocket.send_text(json.dumps(event.as_dict(), default=str))

        tasks = [asyncio.create_task(pump()), asyncio.create_task(heartbeat())]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for task in pending:
            task.cancel()
        for task in done:
            with contextlib.suppress(Exception):
                task.result()
    except WebSocketDisconnect:
        pass
    except Exception:
        with contextlib.suppress(Exception):
            await websocket.close()
    finally:
        subscription.close()
