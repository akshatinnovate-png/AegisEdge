"""Vector clocks and causal delivery.

A hybrid logical clock orders two writes. It does not tell you whether an
operation you just received *depends* on one you have not seen — and applying
a "supersede" before the point it supersedes exists produces a dangling
reference that no amount of retrying fixes.

Vector clocks carry that dependency structure. An operation is delivered only
once its causal predecessors have been, and anything that arrives early waits
in a buffer instead of corrupting state. This is what makes a lossy,
reordering, multi-peer link safe.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from ..core import determinism
from .crdt import Operation


class Ordering(str, Enum):
    BEFORE = "before"
    AFTER = "after"
    CONCURRENT = "concurrent"
    EQUAL = "equal"


@dataclass(frozen=True)
class VectorClock:
    entries: tuple[tuple[str, int], ...] = ()

    @staticmethod
    def of(mapping: dict[str, int]) -> "VectorClock":
        return VectorClock(tuple(sorted((k, v) for k, v in mapping.items() if v)))

    @property
    def mapping(self) -> dict[str, int]:
        return dict(self.entries)

    def tick(self, node_id: str) -> "VectorClock":
        counters = self.mapping
        counters[node_id] = counters.get(node_id, 0) + 1
        return VectorClock.of(counters)

    def merge(self, other: "VectorClock") -> "VectorClock":
        counters = self.mapping
        for node, value in other.mapping.items():
            counters[node] = max(counters.get(node, 0), value)
        return VectorClock.of(counters)

    def compare(self, other: "VectorClock") -> Ordering:
        mine, theirs = self.mapping, other.mapping
        nodes = set(mine) | set(theirs)
        less = any(mine.get(n, 0) < theirs.get(n, 0) for n in nodes)
        greater = any(mine.get(n, 0) > theirs.get(n, 0) for n in nodes)
        if less and greater:
            return Ordering.CONCURRENT
        if less:
            return Ordering.BEFORE
        if greater:
            return Ordering.AFTER
        return Ordering.EQUAL

    def dominates(self, other: "VectorClock") -> bool:
        return self.compare(other) is Ordering.AFTER

    def deliverable(self, incoming: "VectorClock", sender: str) -> bool:
        """Standard causal delivery test for a broadcast from `sender`."""
        mine, theirs = self.mapping, incoming.mapping
        if theirs.get(sender, 0) != mine.get(sender, 0) + 1:
            return False                       # not the next message from that peer
        return all(theirs.get(n, 0) <= mine.get(n, 0) for n in theirs if n != sender)

    def pack(self) -> dict[str, int]:
        return self.mapping

    def __len__(self) -> int:
        return len(self.entries)


@dataclass
class BufferedOp:
    op: Operation
    clock: VectorClock
    sender: str
    received_at: float = field(default_factory=determinism.now)
    attempts: int = 0


class CausalBuffer:
    """Holds operations that arrived before their causal predecessors."""

    def __init__(self, node_id: str, max_hold_s: float = 60.0) -> None:
        self.node_id = node_id
        self.clock = VectorClock()
        self.pending: list[BufferedOp] = []
        self.max_hold_s = max_hold_s
        self.delivered = 0
        self.buffered = 0
        self.released = 0
        self.expired = 0

    def local_event(self) -> VectorClock:
        self.clock = self.clock.tick(self.node_id)
        return self.clock

    def receive(self, op: Operation, clock: VectorClock, sender: str) -> list[Operation]:
        """Accept an op; returns everything now deliverable, in causal order."""
        if not self.clock.deliverable(clock, sender):
            self.pending.append(BufferedOp(op, clock, sender))
            self.buffered += 1
            return []
        deliverable = [self._deliver(op, clock, sender)]
        deliverable.extend(self._drain())
        return deliverable

    def _deliver(self, op: Operation, clock: VectorClock, sender: str) -> Operation:
        self.clock = self.clock.merge(clock)
        self.delivered += 1
        return op

    def _drain(self) -> list[Operation]:
        """Repeatedly release whatever the new state has unblocked."""
        released: list[Operation] = []
        progress = True
        while progress:
            progress = False
            for held in list(self.pending):
                if self.clock.deliverable(held.clock, held.sender):
                    self.pending.remove(held)
                    released.append(self._deliver(held.op, held.clock, held.sender))
                    self.released += 1
                    progress = True
        return released

    def expire(self) -> int:
        """Drop ops whose predecessors never arrived — a peer that vanished."""
        cutoff = determinism.now() - self.max_hold_s
        stale = [h for h in self.pending if h.received_at < cutoff]
        for held in stale:
            self.pending.remove(held)
            self.expired += 1
        return len(stale)

    def snapshot(self) -> dict[str, Any]:
        return {"node": self.node_id, "clock": self.clock.pack(), "peers": len(self.clock),
                "pending": len(self.pending), "delivered": self.delivered,
                "buffered": self.buffered, "released": self.released, "expired": self.expired,
                "oldest_pending_s": round(determinism.now() - min((h.received_at for h in self.pending),
                                                            default=determinism.now()), 2)}
