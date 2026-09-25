"""Durable operation queue.

Everything the node does while offline lands here first, on disk, with an
idempotency key. On reconnect it is replayed in causal order; a delivery that
is retried after an ambiguous failure cannot be applied twice.
"""
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Iterator

from ..core import determinism
from .crdt import Operation


class DurableOpQueue:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.pending: deque[Operation] = deque()
        self.inflight: dict[str, Operation] = {}
        self.acked: set[str] = set()
        self.enqueued = 0
        self.replayed = 0
        self.duplicates_suppressed = 0
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("state") == "acked":
                self.acked.add(record["op_id"])
            elif record.get("op"):
                self.pending.append(Operation.from_dict(record["op"]))
        self.pending = deque(op for op in self.pending if op.op_id not in self.acked)

    def _write(self, record: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def enqueue(self, op: Operation) -> bool:
        if op.op_id in self.acked or any(o.op_id == op.op_id for o in self.pending):
            self.duplicates_suppressed += 1
            return False
        self.pending.append(op)
        self.enqueued += 1
        self._write({"state": "pending", "op_id": op.op_id, "op": op.as_dict(), "ts": determinism.now()})
        return True

    def lease(self, limit: int) -> list[Operation]:
        """Hand out a batch without dropping it — an unacked lease is retried."""
        batch: list[Operation] = []
        while self.pending and len(batch) < limit:
            op = self.pending.popleft()
            self.inflight[op.op_id] = op
            batch.append(op)
        return batch

    def ack(self, op_ids: Iterator[str] | list[str]) -> int:
        count = 0
        for op_id in op_ids:
            if self.inflight.pop(op_id, None) is not None:
                self.acked.add(op_id)
                self._write({"state": "acked", "op_id": op_id, "ts": determinism.now()})
                count += 1
        self.replayed += count
        return count

    def nack(self, ops: list[Operation]) -> None:
        """Return a failed lease to the head of the queue, order preserved."""
        for op in reversed(ops):
            self.inflight.pop(op.op_id, None)
            self.pending.appendleft(op)

    def compact(self) -> None:
        """Rewrite the journal without acked history."""
        rows = [{"state": "pending", "op_id": o.op_id, "op": o.as_dict()} for o in self.pending]
        self.path.write_text("\n".join(json.dumps(r, default=str) for r in rows) + ("\n" if rows else ""),
                             encoding="utf-8")
        self.acked.clear()

    @property
    def depth(self) -> int:
        return len(self.pending) + len(self.inflight)

    def snapshot(self) -> dict[str, int]:
        return {"pending": len(self.pending), "inflight": len(self.inflight),
                "enqueued": self.enqueued, "replayed": self.replayed,
                "duplicates_suppressed": self.duplicates_suppressed}
