"""Crash-safe write-ahead log.

Every mutation lands here, CRC-checked, before it touches the index. An
unclean shutdown replays the log; a torn tail record is truncated rather than
being allowed to half-apply.
"""
from __future__ import annotations

import json
import os
import time
import zlib
from pathlib import Path
from typing import Any, Iterator


class WriteAheadLog:
    def __init__(self, path: Path, fsync: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fsync = fsync
        self._fh = open(self.path, "a+", encoding="utf-8")
        self.appended = 0
        self.torn = 0
        self.checkpoint_lsn = 0
        self.lsn = 0

    def append(self, op: str, body: dict[str, Any]) -> int:
        self.lsn += 1
        record = {"lsn": self.lsn, "ts": time.time(), "op": op, "body": body}
        blob = json.dumps(record, separators=(",", ":"), default=str)
        crc = zlib.crc32(blob.encode("utf-8")) & 0xFFFFFFFF
        self._fh.write(f"{crc:08x} {blob}\n")
        self._fh.flush()
        if self.fsync:
            os.fsync(self._fh.fileno())
        self.appended += 1
        return self.lsn

    def replay(self) -> Iterator[dict[str, Any]]:
        """Yield every intact record, dropping a torn tail."""
        self._fh.seek(0)
        for line in self._fh:
            line = line.rstrip("\n")
            if not line or " " not in line:
                self.torn += 1
                continue
            crc_hex, _, blob = line.partition(" ")
            try:
                if (zlib.crc32(blob.encode("utf-8")) & 0xFFFFFFFF) != int(crc_hex, 16):
                    self.torn += 1
                    continue
                record = json.loads(blob)
            except (ValueError, json.JSONDecodeError):
                self.torn += 1
                continue
            self.lsn = max(self.lsn, record.get("lsn", 0))
            yield record
        self._fh.seek(0, os.SEEK_END)

    def checkpoint(self) -> int:
        """Mark everything up to here as durable in the snapshot."""
        self.checkpoint_lsn = self.lsn
        return self.checkpoint_lsn

    def truncate(self) -> None:
        self._fh.close()
        self.path.write_text("", encoding="utf-8")
        self._fh = open(self.path, "a+", encoding="utf-8")

    def stats(self) -> dict[str, Any]:
        size = self.path.stat().st_size if self.path.exists() else 0
        return {
            "lsn": self.lsn,
            "appended": self.appended,
            "torn": self.torn,
            "checkpoint_lsn": self.checkpoint_lsn,
            "bytes": size,
        }

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:  # pragma: no cover
            pass
