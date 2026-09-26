"""Conflict arbiter.

Resolution ladder, in order: hybrid logical clock, device trust score,
semantic merge (keep both, link them as variants), then a human review queue.
Nothing is silently discarded — an unresolvable conflict becomes a visible
decision rather than a lost memory.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..core.clock import HLC
from ..core import determinism
from .crdt import Operation


class Resolution(str, Enum):
    LOCAL_WINS = "local_wins"
    REMOTE_WINS = "remote_wins"
    MERGED = "merged"
    HUMAN_REVIEW = "human_review"


@dataclass(slots=True)
class ConflictRecord:
    point_id: str
    resolution: Resolution
    rung: str
    local_hlc: str
    remote_hlc: str
    remote_device: str
    detail: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=determinism.now)
    reviewed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {"point_id": self.point_id, "resolution": self.resolution.value, "rung": self.rung,
                "local_hlc": self.local_hlc, "remote_hlc": self.remote_hlc,
                "remote_device": self.remote_device, "detail": self.detail,
                "ts": self.ts, "reviewed": self.reviewed}


class ConflictArbiter:
    SEMANTIC_MERGE_THRESHOLD = 0.82

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self.trust: dict[str, float] = {node_id: 1.0}
        self.records: list[ConflictRecord] = []
        self.review_queue: list[ConflictRecord] = []

    def observe_device(self, device_id: str, delta: float) -> float:
        score = self.trust.get(device_id, 0.75) + delta
        self.trust[device_id] = max(0.0, min(1.0, score))
        return self.trust[device_id]

    def resolve(self, local_clock: HLC, remote_op: Operation, similarity: float | None = None) -> ConflictRecord:
        remote_clock = remote_op.clock
        skew_ms = abs(remote_clock.wall_ms - local_clock.wall_ms)

        # rung 1 — clock, when the two writes are clearly ordered
        if skew_ms >= 250:
            resolution = Resolution.REMOTE_WINS if remote_clock.dominates(local_clock) else Resolution.LOCAL_WINS
            return self._record(remote_op, resolution, "hlc", local_clock, skew_ms=skew_ms)

        # rung 2 — device trust, when the clocks are effectively tied
        local_trust = self.trust.get(self.node_id, 1.0)
        remote_trust = self.trust.get(remote_op.device_id, 0.75)
        if abs(local_trust - remote_trust) >= 0.15:
            resolution = Resolution.REMOTE_WINS if remote_trust > local_trust else Resolution.LOCAL_WINS
            return self._record(remote_op, resolution, "trust", local_clock,
                                local_trust=local_trust, remote_trust=remote_trust)

        # rung 3 — semantic merge: near-identical content is not a real conflict
        if similarity is not None and similarity >= self.SEMANTIC_MERGE_THRESHOLD:
            return self._record(remote_op, Resolution.MERGED, "semantic", local_clock,
                                similarity=round(similarity, 3))

        # rung 4 — a person decides
        record = self._record(remote_op, Resolution.HUMAN_REVIEW, "human", local_clock,
                              similarity=None if similarity is None else round(similarity, 3))
        self.review_queue.append(record)
        return record

    def _record(self, remote_op: Operation, resolution: Resolution, rung: str,
                local_clock: HLC, **detail: Any) -> ConflictRecord:
        record = ConflictRecord(
            point_id=remote_op.point_id, resolution=resolution, rung=rung,
            local_hlc=local_clock.pack(), remote_hlc=remote_op.hlc,
            remote_device=remote_op.device_id, detail=detail,
        )
        self.records.append(record)
        return record

    def review(self, point_id: str, keep: str) -> bool:
        for record in self.review_queue:
            if record.point_id == point_id and not record.reviewed:
                record.reviewed = True
                record.resolution = Resolution.REMOTE_WINS if keep == "remote" else Resolution.LOCAL_WINS
                self.review_queue.remove(record)
                return True
        return False

    def snapshot(self) -> dict[str, Any]:
        by_rung: dict[str, int] = {}
        for record in self.records:
            by_rung[record.rung] = by_rung.get(record.rung, 0) + 1
        return {"total": len(self.records), "by_rung": by_rung,
                "pending_review": len(self.review_queue),
                "trust": {k: round(v, 2) for k, v in self.trust.items()}}
