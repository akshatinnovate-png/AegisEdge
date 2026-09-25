"""The sync engine.

A state machine that assumes it will be interrupted: HOLDING → HANDSHAKE →
DIGEST → PUSH → PULL → CONVERGED, resumable at every step from a durable
cursor. The link dying at 93% resumes at 93%; the session token means a
reconnect is "continue from op 84,213", not a fresh handshake.

Egress runs through the policy engine, not around it: a point the policy
marks local-only is never even offered to the coordinator.
"""
from __future__ import annotations

import asyncio
import time
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from ..core import determinism
from ..core.backoff import DecorrelatedJitter
from ..core.bus import EventBus
from ..core.clock import HybridClock
from ..core.errors import LinkUnavailable
from ..core.metrics import METRICS
from ..core.ratelimit import TokenBucket
from ..memory.schema import MemoryPoint, Sensitivity, SyncClass
from ..memory.store import MemoryStore
from ..policy.engine import PolicyEngine
from ..policy.redaction import RedactionVault
from .conflict import ConflictArbiter, Resolution
from .crdt import OpKind, OpLog, Operation
from .merkle import MerkleTree
from .oracle import ConnectivityOracle, LinkState
from .queue import DurableOpQueue


class SyncState(str, Enum):
    HOLDING = "HOLDING"
    HANDSHAKE = "HANDSHAKE"
    DIGEST = "DIGEST"
    PUSH = "PUSH"
    PULL = "PULL"
    CONVERGED = "CONVERGED"
    BACKOFF = "BACKOFF"


class SyncEngine:
    def __init__(
        self,
        *,
        settings,
        bus: EventBus,
        clock: HybridClock,
        store: MemoryStore,
        policy: PolicyEngine,
        vault: RedactionVault,
        transport,
        oracle: ConnectivityOracle,
    ) -> None:
        self.settings = settings
        self.bus = bus
        self.clock = clock
        self.store = store
        self.policy = policy
        self.vault = vault
        self.transport = transport
        self.oracle = oracle

        self.oplog = OpLog(settings.node_id)
        self.queue = DurableOpQueue(Path(settings.data_dir) / "opqueue.jsonl")
        self.arbiter = ConflictArbiter(settings.node_id)
        self.tree = MerkleTree(settings.sync.merkle_fanout)
        self.bandwidth = TokenBucket(rate=settings.sync.bandwidth_bps / 8,
                                     capacity=settings.sync.bandwidth_bps / 4)
        self.backoff = DecorrelatedJitter(base_ms=250, cap_ms=15_000)

        self.state = SyncState.HOLDING
        self.session: str | None = None
        self.zero_rtt = False
        self.remote_cursor = 0
        self.divergent: list[int] = []
        self.progress = 0.0
        self.last_converged_at: float | None = None
        self.cycles = 0
        self.pushed = 0
        self.pulled = 0
        self.bytes_saved = 0
        self.resumptions = 0
        self.coalesced = 0
        self._lock = asyncio.Lock()
        self._inflight: asyncio.Future | None = None

        oracle.on_restore(self.on_link_restored)

    # -- recording local mutations ----------------------------------------

    def record_local(self, point: MemoryPoint, kind: OpKind = OpKind.UPSERT) -> Operation | None:
        """Turn a local mutation into a queued CRDT op, if policy allows egress."""
        op = Operation(kind=kind, point_id=point.id, hlc=point.hlc or self.clock.now().pack(),
                       device_id=self.settings.node_id, body=self._egress_body(point))
        self.oplog.append(op)
        self.tree.set(point.id, op.hlc)
        if kind is not OpKind.DELETE and not self.policy.may_egress(point):
            self.bus.publish("sync", "egress_denied", level="warn", point_id=point.id,
                             sensitivity=point.sensitivity.value,
                             message=f"<b>{point.id}</b> held on device · {point.sync_class.value}")
            return None
        self.queue.enqueue(op)
        METRICS.gauge("sync.queue_depth", self.queue.depth)
        return op

    def _egress_body(self, point: MemoryPoint) -> dict[str, Any]:
        """Shape what actually leaves: redact, strip, or metadata-only."""
        if point.sync_class is SyncClass.METADATA_ONLY:
            return {"collection": point.collection, "created_at": point.created_at,
                    "sensitivity": point.sensitivity.value, "metadata_only": True,
                    "model_version": point.model_version}
        text = point.text
        redacted: list[str] = []
        if point.sync_class is SyncClass.REDACTED or point.sensitivity is Sensitivity.SENSITIVE:
            text, redacted = self.vault.redact(text)
        payload = {k: v for k, v in point.payload.items() if k not in self.policy.strip_keys}
        return {
            "collection": point.collection, "text": text, "payload": payload,
            "dense": point.dense, "sparse": {str(k): v for k, v in point.sparse.items()},
            "created_at": point.created_at, "confidence": point.confidence,
            "sensitivity": point.sensitivity.value, "model_version": point.model_version,
            "device_id": point.device_id, "redacted": redacted,
            # The handling class travels with the memory. Without it the
            # receiver cannot honour a restriction it was never told about,
            # and re-labels everything it accepts as freely shareable.
            "sync_class": point.sync_class.value,
        }

    # -- the cycle ---------------------------------------------------------

    async def on_link_restored(self) -> None:
        """Fired by the oracle the instant the link comes back."""
        self.bus.publish("sync", "resuming", level="ok",
                         queued=self.queue.depth, session=bool(self.session),
                         message=f"link restored · replaying <b>{self.queue.depth}</b> queued ops")
        await self.reconcile(trigger="link_restored")

    async def reconcile(self, trigger: str = "scheduled") -> dict[str, Any]:
        """Run a cycle, coalescing with one already in flight.

        A manual trigger that lands mid-cycle used to return a bare
        "already_running", which tells the caller nothing and looks like a
        failure. Instead the caller joins the running cycle and receives its
        result — the same answer it would have got, without a second pass over
        the same operations.
        """
        if self._inflight is not None and not self._inflight.done():
            self.coalesced += 1
            METRICS.incr("sync.coalesced")
            result = await asyncio.shield(self._inflight)
            return {**result, "coalesced_with": "in-flight cycle"}

        loop = asyncio.get_running_loop()
        self._inflight = loop.create_future()
        try:
            async with self._lock:
                result = await self._reconcile(trigger)
            if not self._inflight.done():
                self._inflight.set_result(result)
            return result
        except Exception as exc:
            if self._inflight is not None and not self._inflight.done():
                self._inflight.set_exception(exc)
            raise
        finally:
            self._inflight = None

    async def _reconcile(self, trigger: str) -> dict[str, Any]:
        t0 = determinism.monotonic()
        self.cycles += 1
        self.progress = 0.0
        pushed = pulled = conflicts = 0
        try:
            # 1. handshake — resumed sessions skip the full round trip
            self._set_state(SyncState.HANDSHAKE)
            info = await self.transport.handshake(self.settings.node_id, self.session)
            self.session = info.get("session")
            self.zero_rtt = bool(info.get("zero_rtt"))
            if info.get("resumed"):
                self.resumptions += 1
            self.remote_cursor = max(self.remote_cursor, int(info.get("their_cursor", 0)))
            self.progress = 0.15

            # 2. digest — find the few buckets that actually differ
            self._set_state(SyncState.DIGEST)
            remote_digest = await self.transport.digest()
            self.divergent = self.tree.divergent(remote_digest)
            self.bytes_saved += self.tree.bytes_saved(self.divergent)
            self.progress = 0.35

            # 3. push — durable queue drains in causal order
            self._set_state(SyncState.PUSH)
            pushed = await self._push()
            self.progress = 0.7

            # 4. pull — fleet knowledge relevant to this device
            self._set_state(SyncState.PULL)
            pulled, conflicts = await self._pull()
            self.progress = 1.0

            self._set_state(SyncState.CONVERGED)
            self.last_converged_at = determinism.now()
            self.backoff.reset()
            self.arbiter.observe_device(self.settings.node_id, +0.01)
            duration = (determinism.monotonic() - t0) * 1000
            METRICS.observe("sync.cycle_ms", duration)
            self.bus.publish(
                "sync", "converged", level="ok", trigger=trigger,
                pushed=pushed, pulled=pulled, conflicts=conflicts,
                divergent=len(self.divergent), duration_ms=round(duration, 1),
                cursor=self.oplog.cursor, resumed=self.zero_rtt,
                message=(f"converged · ↑{pushed} ↓{pulled} · {len(self.divergent)} divergent "
                         f"ranges · {duration:.0f} ms"),
            )
            return {"state": self.state.value, "pushed": pushed, "pulled": pulled,
                    "conflicts": conflicts, "divergent": len(self.divergent),
                    "duration_ms": round(duration, 1), "resumed": self.zero_rtt}
        except LinkUnavailable as exc:
            self._set_state(SyncState.BACKOFF)
            delay = self.backoff.next_delay_ms()
            METRICS.incr("sync.interrupted")
            self.bus.publish("sync", "interrupted", level="warn", error=str(exc),
                             retry_in_ms=delay, queued=self.queue.depth,
                             progress=round(self.progress, 2),
                             message=(f"sync interrupted at <b>{self.progress:.0%}</b> — "
                                      f"resuming from cursor, retry in {delay:.0f} ms"))
            return {"state": self.state.value, "error": str(exc), "retry_in_ms": delay,
                    "progress": self.progress}

    async def _push(self) -> int:
        sent = 0
        while self.queue.pending:
            batch = self.queue.lease(min(self.settings.sync.batch_ops, self.policy.egress_batch))
            if not batch:
                break
            await self.bandwidth.take(min(len(batch) * 1800, self.bandwidth.capacity))
            try:
                result = await self.transport.push(batch)
            except Exception:
                self.queue.nack(batch)               # nothing is lost on a failed lease
                raise
            self.queue.ack(result.get("accepted", []))
            rejected = [op for op in batch if op.op_id in set(result.get("rejected", []))]
            self.queue.ack([op.op_id for op in rejected])   # cloud already had newer
            sent += len(result.get("accepted", []))
            self.pushed += len(result.get("accepted", []))
            METRICS.gauge("sync.queue_depth", self.queue.depth)
        return sent

    async def _pull(self) -> tuple[int, int]:
        applied = conflicts = 0
        while True:
            page = await self.transport.pull(self.remote_cursor, self.settings.sync.batch_ops)
            ops = [Operation.from_dict(o) for o in page.get("ops", [])]
            if not ops:
                self.remote_cursor = int(page.get("cursor", self.remote_cursor))
                break
            mine = self.settings.node_id
            foreign = [op for op in ops if op.device_id != mine]
            accepted, conflicted, _dropped = self.oplog.merge(foreign)
            for op in accepted:
                self.clock.observe(op.clock)
                await self._materialize(op)
                applied += 1
            for op in conflicted:
                conflicts += 1
                await self._arbitrate(op)
            self.remote_cursor = int(page.get("cursor", self.remote_cursor))
            self.pulled += applied
            if not page.get("remaining"):
                break
        return applied, conflicts

    async def _materialize(self, op: Operation) -> None:
        """Turn a remote op into local state."""
        if op.kind is OpKind.DELETE:
            self.store.delete(op.point_id)
            self.tree.drop(op.point_id)
            return
        body = op.body or {}
        if body.get("metadata_only"):
            return
        point = MemoryPoint(
            id=op.point_id, collection=body.get("collection", "semantic"),
            text=body.get("text", ""), dense=body.get("dense", []),
            sparse={int(k): float(v) for k, v in (body.get("sparse") or {}).items()},
            payload=body.get("payload", {}), confidence=float(body.get("confidence", 0.9)),
            sensitivity=Sensitivity(body.get("sensitivity", "internal")),
            # Never less restrictive than the sender said. Hard-coding FULL
            # here meant a memory marked for redaction at its origin arrived
            # freely shareable, so the next hop made its decisions on a class
            # the memory never had. Older peers omit the field; absent it, the
            # sender's own egress rules already applied, so FULL is the right
            # floor and the local policy may still tighten it.
            sync_class=SyncClass.strictest(
                SyncClass(body.get("sync_class", SyncClass.FULL.value)),
                SyncClass.FULL),
            model_version=body.get("model_version", ""),
            device_id=body.get("device_id", op.device_id), hlc=op.hlc,
            source="fleet", created_at=body.get("created_at", determinism.now()),
        )
        if not point.dense:
            point.dense = self.store.embedder.embed_sync([point.text])[0].tolist()
        self.store.apply_remote(point)
        self.tree.set(point.id, op.hlc)

    async def _arbitrate(self, op: Operation) -> None:
        local_point = self.store.points.get(op.point_id)
        local_clock = self.oplog.heads.get(op.point_id) or self.clock.now()
        similarity = None
        if local_point and local_point.dense and op.body.get("dense"):
            a = np.asarray(local_point.dense, dtype=np.float32)
            b = np.asarray(op.body["dense"], dtype=np.float32)
            denominator = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
            similarity = float(a @ b / denominator)
        record = self.arbiter.resolve(local_clock, op, similarity)
        if record.resolution is Resolution.REMOTE_WINS:
            await self._materialize(op)
        elif record.resolution is Resolution.MERGED and local_point:
            await self._materialize(op)
            self.store.supersede(local_point.id, op.point_id, reason="semantic_merge")
        self.bus.publish(
            "sync", "conflict", level="warn", point_id=op.point_id,
            resolution=record.resolution.value, rung=record.rung,
            message=(f"conflict on <b>{op.point_id}</b> → {record.resolution.value} "
                     f"(rung: {record.rung})"),
        )

    def _set_state(self, state: SyncState) -> None:
        self.state = state
        self.bus.publish("sync", "state", state=state.value, progress=round(self.progress, 2))

    # -- loop --------------------------------------------------------------

    async def run(self) -> None:
        while True:
            interval = self.settings.sync.interval_s
            if self.oracle.state is LinkState.OFFLINE:
                self.state = SyncState.HOLDING
                await asyncio.sleep(1.0)
                continue
            if self.state is SyncState.BACKOFF:
                await asyncio.sleep(self.backoff.next_delay_ms() / 1000.0)
            await self.reconcile()
            await asyncio.sleep(interval)

    # -- reporting ---------------------------------------------------------

    def status(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "link": self.oracle.state.value,
            "queued": self.queue.depth,
            "divergent": len(self.divergent),
            "conflicts": len(self.arbiter.records),
            "pending_review": len(self.arbiter.review_queue),
            "progress": round(self.progress, 3),
            "cursor": self.oplog.cursor,
            "remote_cursor": self.remote_cursor,
            "session": self.session,
            "zero_rtt_resumed": self.zero_rtt,
            "resumptions": self.resumptions,
            "coalesced": self.coalesced,
            "cycles": self.cycles,
            "pushed": self.pushed,
            "pulled": self.pulled,
            "bytes_saved": self.bytes_saved,
            "last_converged_at": self.last_converged_at,
            "oplog": self.oplog.snapshot(),
            "queue": self.queue.snapshot(),
            "arbiter": self.arbiter.snapshot(),
            "transport": self.transport.snapshot() if hasattr(self.transport, "snapshot") else {},
        }
