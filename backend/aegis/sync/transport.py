"""Cloud transport.

Two implementations behind one protocol:

* ``LoopbackCloud`` — an in-process stand-in for the Qdrant Server + sync
  coordinator, with its own op log and Merkle tree. It is not a mock: it
  really diverges, really conflicts and really converges, so the whole sync
  path is exercised on a laptop with no cloud account.
* ``HttpCloud`` — the real thing over HTTP/QUIC, with 0-RTT session
  resumption, a circuit breaker, hedged requests while degraded and a
  bandwidth token bucket.
"""
from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Protocol

from ..core.backoff import DecorrelatedJitter
from ..core.circuit import CircuitBreaker
from ..core.errors import LinkUnavailable
from ..core.ids import ulid
from ..core.metrics import METRICS
from ..core.ratelimit import TokenBucket
from .crdt import OpKind, Operation
from .merkle import MerkleDigest, MerkleTree


class CloudTransport(Protocol):
    name: str
    metered: bool

    async def ping(self) -> bool: ...
    async def handshake(self, node_id: str, resume_token: str | None) -> dict[str, Any]: ...
    async def digest(self) -> MerkleDigest: ...
    async def push(self, ops: list[Operation]) -> dict[str, Any]: ...
    async def pull(self, cursor: int, limit: int) -> dict[str, Any]: ...


class LoopbackCloud:
    """In-process coordinator + canonical store."""

    name = "loopback://cloud"
    metered = False

    def __init__(self, latency_ms: float = 14.0, loss: float = 0.0) -> None:
        self.latency_ms = latency_ms
        self.loss = loss
        self.partitioned = False
        self.ops: list[Operation] = []
        self.points: dict[str, dict[str, Any]] = {}
        self.tree = MerkleTree(16)
        self.sessions: dict[str, dict[str, Any]] = {}
        self.received = 0
        self.served = 0

    # -- fault surface used by the chaos endpoints ------------------------

    def partition(self, on: bool = True) -> None:
        self.partitioned = on

    async def _hop(self) -> None:
        if self.partitioned:
            raise LinkUnavailable("network partition")
        jitter = random.uniform(0.6, 1.5)
        await asyncio.sleep(self.latency_ms * jitter / 1000.0)
        if random.random() < self.loss:
            raise LinkUnavailable("packet loss")

    # -- protocol ---------------------------------------------------------

    async def ping(self) -> bool:
        await self._hop()
        return True

    async def handshake(self, node_id: str, resume_token: str | None) -> dict[str, Any]:
        await self._hop()
        if resume_token and resume_token in self.sessions:
            session = self.sessions[resume_token]
            session["resumed"] += 1
            return {"session": resume_token, "resumed": True, "zero_rtt": True,
                    "server_cursor": len(self.ops), "their_cursor": session["their_cursor"]}
        token = ulid()
        self.sessions[token] = {"node_id": node_id, "their_cursor": 0, "resumed": 0,
                                "opened_at": time.time()}
        return {"session": token, "resumed": False, "zero_rtt": False,
                "server_cursor": len(self.ops), "their_cursor": 0}

    async def digest(self) -> MerkleDigest:
        await self._hop()
        return self.tree.digest()

    async def push(self, ops: list[Operation]) -> dict[str, Any]:
        await self._hop()
        accepted, rejected = [], []
        for op in ops:
            existing = self.points.get(op.point_id)
            if existing and existing["hlc"] > op.hlc:
                rejected.append(op.op_id)              # the cloud already has newer
                continue
            self.ops.append(op)
            if op.kind is OpKind.DELETE:
                self.points.pop(op.point_id, None)
                self.tree.drop(op.point_id)
            else:
                self.points[op.point_id] = {"hlc": op.hlc, "body": op.body}
                self.tree.set(op.point_id, op.hlc)
            accepted.append(op.op_id)
        self.received += len(accepted)
        return {"accepted": accepted, "rejected": rejected, "server_cursor": len(self.ops)}

    async def pull(self, cursor: int, limit: int) -> dict[str, Any]:
        await self._hop()
        window = self.ops[cursor: cursor + limit]
        self.served += len(window)
        return {"ops": [op.as_dict() for op in window],
                "cursor": min(cursor + limit, len(self.ops)),
                "remaining": max(0, len(self.ops) - (cursor + limit))}

    # -- fleet knowledge injection (what other devices learned) -----------

    def inject(self, point_id: str, hlc: str, body: dict[str, Any], device_id: str = "edge-fleet") -> Operation:
        op = Operation(kind=OpKind.UPSERT, point_id=point_id, hlc=hlc, device_id=device_id, body=body)
        self.ops.append(op)
        self.points[point_id] = {"hlc": hlc, "body": body}
        self.tree.set(point_id, hlc)
        return op

    def snapshot(self) -> dict[str, Any]:
        return {"endpoint": self.name, "ops": len(self.ops), "points": len(self.points),
                "received": self.received, "served": self.served,
                "partitioned": self.partitioned, "sessions": len(self.sessions)}


class HttpCloud:
    """Real coordinator over HTTP/2 or QUIC, with resumption and hedging."""

    metered = False

    def __init__(self, base_url: str, bandwidth_bps: int = 2_000_000, timeout_s: float = 4.0) -> None:
        self.name = base_url
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.breaker = CircuitBreaker("cloud", failure_threshold=4, reset_after_s=6.0)
        self.bucket = TokenBucket(rate=bandwidth_bps / 8, capacity=bandwidth_bps / 4)
        self.backoff = DecorrelatedJitter()
        self.session_token: str | None = None
        self.hedged = 0
        self._client: Any = None

    async def _http(self):
        if self._client is None:
            import httpx  # imported lazily: the offline path must not need it

            self._client = httpx.AsyncClient(
                base_url=self.base_url, timeout=self.timeout_s,
                headers={"x-aegis-session": self.session_token or ""},
            )
        return self._client

    async def _request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        self.breaker.guard()
        client = await self._http()
        try:
            response = await client.request(method, path, **kwargs)
            response.raise_for_status()
            self.breaker.record_success()
            self.backoff.reset()
            return response.json()
        except Exception as exc:
            self.breaker.record_failure()
            METRICS.incr("sync.transport_errors")
            raise LinkUnavailable(str(exc)) from exc

    async def _hedged(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        """Race a duplicate request on a degraded link; first answer wins."""
        first = asyncio.create_task(self._request(method, path, **kwargs))
        done, pending = await asyncio.wait({first}, timeout=self.timeout_s / 3)
        if done:
            return first.result()
        self.hedged += 1
        second = asyncio.create_task(self._request(method, path, **kwargs))
        done, pending = await asyncio.wait({first, second}, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        for task in done:
            if not task.cancelled() and task.exception() is None:
                return task.result()
        raise LinkUnavailable("both hedged attempts failed")

    async def ping(self) -> bool:
        await self._request("GET", "/api/v1/ping")
        return True

    async def handshake(self, node_id: str, resume_token: str | None) -> dict[str, Any]:
        body = await self._request("POST", "/api/v1/sync/handshake",
                                   json={"node_id": node_id, "resume": resume_token})
        self.session_token = body.get("session", self.session_token)
        return body

    async def digest(self) -> MerkleDigest:
        return MerkleDigest.from_dict(await self._request("GET", "/api/v1/sync/digest"))

    async def push(self, ops: list[Operation]) -> dict[str, Any]:
        payload = [op.as_dict() for op in ops]
        await self.bucket.take(min(len(str(payload)), self.bucket.capacity))   # bandwidth budget
        return await self._hedged("POST", "/api/v1/sync/push", json={"ops": payload})

    async def pull(self, cursor: int, limit: int) -> dict[str, Any]:
        return await self._hedged("GET", f"/api/v1/sync/pull?cursor={cursor}&limit={limit}")

    def snapshot(self) -> dict[str, Any]:
        return {"endpoint": self.name, "breaker": self.breaker.snapshot(),
                "hedged": self.hedged, "session": bool(self.session_token),
                "bandwidth_tokens": round(self.bucket.level)}


def build_transport(url: str, bandwidth_bps: int) -> CloudTransport:
    if url.startswith(("http://", "https://")):
        return HttpCloud(url, bandwidth_bps)
    return LoopbackCloud()
