"""CRDT operation log.

Two devices that were partitioned must converge to the same state without a
coordinator deciding who was right. Every mutation is an operation stamped
with a hybrid logical clock; upserts are last-writer-wins per point, deletes
are observed-remove tombstones. Merge is commutative, associative and
idempotent — which is what makes replay-on-reconnect safe.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from ..core.clock import HLC
from ..core.ids import ulid


class OpKind(str, Enum):
    UPSERT = "upsert"
    DELETE = "delete"
    SUPERSEDE = "supersede"


@dataclass(slots=True)
class Operation:
    op_id: str = field(default_factory=ulid)
    kind: OpKind = OpKind.UPSERT
    point_id: str = ""
    hlc: str = ""
    device_id: str = ""
    body: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    @property
    def clock(self) -> HLC:
        return HLC.parse(self.hlc) if self.hlc else HLC(0, 0, self.device_id)

    def digest(self) -> str:
        blob = json.dumps({"k": self.kind.value, "p": self.point_id, "h": self.hlc},
                          sort_keys=True, separators=(",", ":"))
        return hashlib.blake2b(blob.encode("utf-8"), digest_size=8).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {"op_id": self.op_id, "kind": self.kind.value, "point_id": self.point_id,
                "hlc": self.hlc, "device_id": self.device_id, "body": self.body, "ts": self.ts}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Operation":
        return Operation(op_id=d["op_id"], kind=OpKind(d["kind"]), point_id=d["point_id"],
                         hlc=d.get("hlc", ""), device_id=d.get("device_id", ""),
                         body=d.get("body", {}), ts=d.get("ts", time.time()))


class OpLog:
    """Append-only causal log with LWW-per-point resolution and OR-Set deletes."""

    def __init__(self, device_id: str) -> None:
        self.device_id = device_id
        self.ops: list[Operation] = []
        self.seen: set[str] = set()                 # idempotency: op ids already applied
        self.heads: dict[str, HLC] = {}             # point_id -> winning clock
        self.tombstones: dict[str, HLC] = {}
        self.applied = 0
        self.rejected_stale = 0

    def append(self, op: Operation) -> bool:
        if op.op_id in self.seen:
            return False                            # replay is a no-op, by construction
        self.seen.add(op.op_id)
        self.ops.append(op)
        return self._resolve(op)

    def _resolve(self, op: Operation) -> bool:
        current = self.heads.get(op.point_id)
        clock = op.clock
        if current is not None and not clock.dominates(current):
            self.rejected_stale += 1
            return False                            # an older write never overwrites a newer one
        self.heads[op.point_id] = clock
        if op.kind is OpKind.DELETE:
            self.tombstones[op.point_id] = clock
        else:
            self.tombstones.pop(op.point_id, None)
        self.applied += 1
        return True

    CONCURRENCY_WINDOW_MS = 1000

    def merge(self, remote: Iterable[Operation]) -> tuple[list[Operation], list[Operation], list[Operation]]:
        """Merge peer operations.

        Returns (accepted, conflicted, dropped). An op that loses cleanly on
        the clock is *dropped* — convergence, not a conflict. An op that lost
        but was written concurrently (within the skew window) is a genuine
        conflict and goes to the arbiter.
        """
        accepted: list[Operation] = []
        conflicted: list[Operation] = []
        dropped: list[Operation] = []
        for op in remote:
            if op.op_id in self.seen:
                continue
            current = self.heads.get(op.point_id)
            if current is not None and not op.clock.dominates(current):
                if abs(op.clock.wall_ms - current.wall_ms) < self.CONCURRENCY_WINDOW_MS:
                    conflicted.append(op)
                else:
                    self.seen.add(op.op_id)
                    self.rejected_stale += 1
                    dropped.append(op)
                continue
            self.append(op)
            accepted.append(op)
        return accepted, conflicted, dropped

    def since(self, cursor: int, limit: int = 256) -> list[Operation]:
        return self.ops[cursor: cursor + limit]

    @property
    def cursor(self) -> int:
        return len(self.ops)

    def snapshot(self) -> dict[str, Any]:
        return {"ops": len(self.ops), "applied": self.applied, "tombstones": len(self.tombstones),
                "rejected_stale": self.rejected_stale, "points_tracked": len(self.heads)}
