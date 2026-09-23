"""Tamper-evident audit log.

Hash-chained and append-only: altering entry *n* invalidates every entry after
it, so a device that has been tampered with cannot quietly rewrite what it did
with the data it held.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

GENESIS = "0" * 64


@dataclass(slots=True)
class AuditEntry:
    seq: int
    ts: float
    action: str
    subject: str
    detail: dict[str, Any] = field(default_factory=dict)
    prev_hash: str = GENESIS
    hash: str = ""

    def compute_hash(self) -> str:
        body = json.dumps(
            {"seq": self.seq, "ts": round(self.ts, 6), "action": self.action,
             "subject": self.subject, "detail": self.detail, "prev": self.prev_hash},
            sort_keys=True, separators=(",", ":"), default=str,
        )
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {"seq": self.seq, "ts": self.ts, "action": self.action, "subject": self.subject,
                "detail": self.detail, "hash": self.hash[:16], "prev_hash": self.prev_hash[:16]}


class AuditLog:
    def __init__(self, path: Path | None = None, keep: int = 2000) -> None:
        self.path = Path(path) if path else None
        self.entries: list[AuditEntry] = []
        self.keep = keep
        self._tip = GENESIS
        self._seq = 0
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, action: str, subject: str, **detail: Any) -> AuditEntry:
        self._seq += 1
        entry = AuditEntry(seq=self._seq, ts=time.time(), action=action,
                           subject=subject, detail=detail, prev_hash=self._tip)
        entry.hash = entry.compute_hash()
        self._tip = entry.hash
        self.entries.append(entry)
        if len(self.entries) > self.keep:
            del self.entries[: len(self.entries) - self.keep]
        if self.path:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry.as_dict(), default=str) + "\n")
        return entry

    def verify(self) -> tuple[bool, int | None]:
        """Walk the chain; returns (intact, first_broken_seq)."""
        prev = self.entries[0].prev_hash if self.entries else GENESIS
        for entry in self.entries:
            if entry.prev_hash != prev or entry.compute_hash() != entry.hash:
                return False, entry.seq
            prev = entry.hash
        return True, None

    def tail(self, limit: int = 50) -> list[dict[str, Any]]:
        return [e.as_dict() for e in self.entries[-limit:]]

    def snapshot(self) -> dict[str, Any]:
        intact, broken = self.verify()
        return {"entries": self._seq, "resident": len(self.entries),
                "chain_intact": intact, "broken_at": broken, "tip": self._tip[:16]}
