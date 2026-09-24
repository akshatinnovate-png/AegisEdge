"""Vector storage: resident rows for the working set, memmap for the cold tail.

The cold tier's whole purpose is that its vectors are *not* in RAM. Keeping a
full-precision copy resident "for rescoring" quietly defeats that — the
compression ratio becomes a slide, not a fact.

So cold vectors are evicted to a memory-mapped file on disk. Only the 1-bit or
PQ codes stay resident; rescoring gathers the shortlist's rows back from the
map, which is a handful of page-ins, not a corpus-sized read.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .growable import GrowableMatrix


class VectorStorage:
    GROWTH = 4096

    @property
    def resident(self) -> np.ndarray:
        return self.rows.view

    def __init__(self, dim: int, path: Path | None = None) -> None:
        self.dim = dim
        self.path = Path(path) if path else None
        self.rows = GrowableMatrix(dim)
        self.row_of: dict[str, int] = {}
        self.ids: list[str] = []
        self.free_rows: list[int] = []

        self._map: np.memmap | None = None
        self._map_rows = 0
        self.cold_row_of: dict[str, int] = {}
        self.page_ins = 0
        self.evictions = 0

    # -- resident ---------------------------------------------------------

    def put(self, point_id: str, vector: np.ndarray) -> int:
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector)) or 1.0
        vector = vector / norm
        row = self.row_of.get(point_id)
        if row is not None:
            self.resident[row] = vector
            return row
        if self.free_rows:
            row = self.free_rows.pop()
            self.rows[row] = vector
        else:
            row = self.rows.append(vector)
        self.row_of[point_id] = row
        self.ids.append(point_id)
        self.cold_row_of.pop(point_id, None)
        return row

    def get(self, point_id: str) -> np.ndarray | None:
        row = self.row_of.get(point_id)
        if row is not None:
            return self.resident[row]
        cold = self.cold_row_of.get(point_id)
        if cold is not None and self._map is not None:
            self.page_ins += 1
            return np.asarray(self._map[cold], dtype=np.float32)
        return None

    def gather(self, point_ids: list[str]) -> np.ndarray:
        """Fetch several vectors, paging in cold rows only for those asked for."""
        out = np.zeros((len(point_ids), self.dim), dtype=np.float32)
        cold: list[tuple[int, int]] = []
        for i, point_id in enumerate(point_ids):
            row = self.row_of.get(point_id)
            if row is not None:
                out[i] = self.resident[row]
            elif point_id in self.cold_row_of:
                cold.append((i, self.cold_row_of[point_id]))
        if cold and self._map is not None:
            rows = [r for _, r in cold]
            self.page_ins += len(rows)
            block = np.asarray(self._map[rows], dtype=np.float32)
            for (slot, _), vector in zip(cold, block):
                out[slot] = vector
        return out

    def drop(self, point_id: str) -> bool:
        row = self.row_of.pop(point_id, None)
        if row is not None:
            self.free_rows.append(row)                # reuse the slot, don't reshuffle
            self.rows[row] = 0.0
        cold = self.cold_row_of.pop(point_id, None)
        if point_id in self.ids:
            self.ids.remove(point_id)
        return row is not None or cold is not None

    # -- cold tier --------------------------------------------------------

    def _ensure_map(self, rows: int) -> None:
        if self.path is None:
            return
        if self._map is not None and self._map_rows >= rows:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new_rows = max(rows, self._map_rows + self.GROWTH)
        existing = None
        if self._map is not None:
            existing = np.asarray(self._map[: self._map_rows], dtype=np.float32).copy()
            self._map.flush()
            del self._map
        self._map = np.memmap(self.path, dtype=np.float32, mode="w+", shape=(new_rows, self.dim))
        if existing is not None and len(existing):
            self._map[: len(existing)] = existing
        self._map_rows = new_rows

    def evict(self, point_id: str) -> bool:
        """Move a vector out of RAM and onto disk."""
        if self.path is None:
            return False
        row = self.row_of.get(point_id)
        if row is None:
            return point_id in self.cold_row_of
        vector = self.resident[row].copy()
        cold_row = len(self.cold_row_of)
        self._ensure_map(cold_row + 1)
        if self._map is None:
            return False
        self._map[cold_row] = vector
        self.cold_row_of[point_id] = cold_row
        self.row_of.pop(point_id, None)
        self.free_rows.append(row)
        self.rows[row] = 0.0
        self.evictions += 1
        return True

    def promote(self, point_id: str) -> bool:
        """Bring a cold vector back into RAM."""
        cold = self.cold_row_of.get(point_id)
        if cold is None or self._map is None:
            return False
        self.put(point_id, np.asarray(self._map[cold], dtype=np.float32))
        return True

    def flush(self) -> None:
        if self._map is not None:
            self._map.flush()

    # -- reporting --------------------------------------------------------

    @property
    def resident_count(self) -> int:
        return len(self.row_of)

    @property
    def cold_count(self) -> int:
        return len(self.cold_row_of)

    def snapshot(self) -> dict[str, object]:
        on_disk = 0
        if self.path is not None and self.path.exists():
            on_disk = self.path.stat().st_size
        return {
            "resident": self.resident_count,
            "cold_on_disk": self.cold_count,
            "resident_bytes": int(self.resident_count * self.dim * 4),
            "disk_bytes": on_disk,
            "page_ins": self.page_ins,
            "evictions": self.evictions,
            "free_rows": len(self.free_rows),
            "buffer": self.rows.snapshot(),
        }
