"""Mesh over HTTP: two real devices, no cloud between them.

`MeshLink` dispatches into a dict of in-process agents. That is exactly right
for the test suite and for a fifty-peer fleet simulated in one process, and
useless for two devices sitting on a bench — which is the case the problem
statement actually cares about, and the one a person can watch happen.

Nothing above this had to change, because the seam was already in the right
place: `call(sender, target, method, payload) -> dict` is an RPC with the
transport left out, and its own docstring says a radio or mDNS transport swaps
in behind it. This is that transport, over HTTP.

The peer on the other end is a real node process. It receives the call on
`POST /api/v1/mesh/exchange` and hands it straight to its own `GossipAgent`,
so both sides run the same anti-entropy, the same IBLT reconciliation, the
same causal delivery and the same policy filtering they run in-process. There
is no second code path to keep honest.

Offline is a first-class state here rather than an error to be surprised by.
A device with its link pulled refuses to send and says so, which is what every
caller above already expects from a partitioned peer.
"""
from __future__ import annotations

import asyncio
import random
import time
from typing import Any

import httpx

EXCHANGE_PATH = "/api/v1/mesh/exchange"


class HttpMeshLink:
    """The `MeshLink` contract, carried over HTTP to peer node processes."""

    def __init__(self, timeout_s: float = 6.0, latency_ms: float = 0.0,
                 loss: float = 0.0) -> None:
        self.endpoints: dict[str, str] = {}
        self.timeout_s = float(timeout_s)
        self.latency_ms = float(latency_ms)     # optional synthetic delay
        self.loss = float(loss)
        self.partitions: set[tuple[str, str]] = set()
        self.messages = 0
        self.dropped = 0
        self.offline = False
        self.node_id: str | None = None
        self._client: httpx.AsyncClient | None = None
        self.last_error: str | None = None

    # -- membership -------------------------------------------------------

    def join(self, agent: Any) -> None:
        """Remember who we are. There is no local registry to join."""
        self.node_id = agent.node_id

    def register(self, node_id: str, endpoint: str) -> None:
        if endpoint:
            self.endpoints[node_id] = endpoint.rstrip("/")

    def forget(self, node_id: str) -> None:
        self.endpoints.pop(node_id, None)

    # -- fault surface ----------------------------------------------------

    def partition(self, a: str, b: str, on: bool = True) -> None:
        key = tuple(sorted((a, b)))
        self.partitions.add(key) if on else self.partitions.discard(key)

    def reachable(self, a: str, b: str) -> bool:
        return tuple(sorted((a, b))) not in self.partitions

    def set_offline(self, on: bool = True) -> bool:
        """Pull the radio. Every send fails until it is put back."""
        self.offline = bool(on)
        return self.offline

    # -- transport --------------------------------------------------------

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout_s)
        return self._client

    async def call(self, sender: str, target: str, method: str,
                   payload: dict[str, Any]) -> dict[str, Any]:
        self.messages += 1
        if self.offline:
            self.dropped += 1
            raise ConnectionError(f"{sender} is offline")
        endpoint = self.endpoints.get(target)
        if not endpoint or not self.reachable(sender, target):
            self.dropped += 1
            raise ConnectionError(f"{target} unreachable from {sender}")
        if self.latency_ms:
            await asyncio.sleep(self.latency_ms * random.uniform(0.7, 1.4) / 1000.0)
        if self.loss and random.random() < self.loss:
            self.dropped += 1
            raise ConnectionError("packet lost")

        client = await self._http()
        started = time.perf_counter()
        try:
            response = await client.post(
                endpoint + EXCHANGE_PATH,
                json={"sender": sender, "method": method, "payload": payload})
            response.raise_for_status()
            self.last_error = None
            return response.json()
        except Exception as exc:
            # Every caller above already knows how to survive an unreachable
            # peer. Presenting a transport failure as anything else would make
            # a normal condition look like a defect.
            self.dropped += 1
            self.last_error = f"{type(exc).__name__}: {str(exc)[:120]}"
            raise ConnectionError(f"{target} unreachable: {self.last_error}") from exc
        finally:
            self.rtt_ms = (time.perf_counter() - started) * 1000

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "transport": "http", "node_id": self.node_id,
            "endpoints": dict(self.endpoints), "offline": self.offline,
            "messages": self.messages, "dropped": self.dropped,
            "partitions": ["|".join(p) for p in self.partitions],
            "last_error": self.last_error,
        }
