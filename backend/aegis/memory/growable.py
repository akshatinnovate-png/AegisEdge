"""A row-appendable float32 matrix.

`np.vstack([matrix, row])` allocates a new array and copies every existing row,
so building an index one point at a time costs O(n) per insert and O(n²)
overall. At 50k points and 256 dimensions that is a 51 MB copy on *every*
write — the ingest rate visibly decays as the corpus grows, which a stress run
surfaces immediately.

This grows capacity geometrically instead, so appends are amortised O(1), and
exposes the filled region as a **view** rather than a copy so every existing
`matrix @ query` stays a single BLAS call over contiguous memory.

Removal is swap-with-last rather than `np.delete`, which is also O(1); callers
that map row index to identity are handed the moved row so they can fix their
own bookkeeping.
"""
from __future__ import annotations

import numpy as np


class GrowableMatrix:
    __slots__ = ("dim", "_buffer", "_rows", "growth", "reallocations", "appends")

    def __init__(self, dim: int, capacity: int = 1024, growth: float = 2.0) -> None:
        self.dim = dim
        self.growth = growth
        self._buffer = np.zeros((max(capacity, 1), dim), dtype=np.float32)
        self._rows = 0
        self.reallocations = 0
        self.appends = 0

    # -- shape ------------------------------------------------------------

    def __len__(self) -> int:
        return self._rows

    @property
    def capacity(self) -> int:
        return int(self._buffer.shape[0])

    @property
    def view(self) -> np.ndarray:
        """The filled rows, as a view — never a copy."""
        return self._buffer[: self._rows]

    def __getitem__(self, index):
        return self.view[index]

    def __setitem__(self, index, value) -> None:
        self.view[index] = value

    # -- writes -----------------------------------------------------------

    def _reserve(self, needed: int) -> None:
        if needed <= self.capacity:
            return
        capacity = self.capacity
        while capacity < needed:
            capacity = max(int(capacity * self.growth), capacity + 1)
        grown = np.zeros((capacity, self.dim), dtype=np.float32)
        grown[: self._rows] = self.view
        self._buffer = grown
        self.reallocations += 1

    def append(self, vector: np.ndarray) -> int:
        """Add a row, returning its index."""
        self._reserve(self._rows + 1)
        self._buffer[self._rows] = vector
        self._rows += 1
        self.appends += 1
        return self._rows - 1

    def swap_remove(self, index: int) -> int | None:
        """Delete a row in O(1).

        The last row is moved into the hole and its old index is returned so
        the caller can repoint whatever identity it kept for it; None means the
        removed row was the last one and nothing moved.
        """
        if not 0 <= index < self._rows:
            return None
        last = self._rows - 1
        moved: int | None = None
        if index != last:
            self._buffer[index] = self._buffer[last]
            moved = last
        self._buffer[last] = 0.0
        self._rows -= 1
        return moved

    def clear(self) -> None:
        self._rows = 0

    # -- reporting --------------------------------------------------------

    @property
    def nbytes(self) -> int:
        return int(self._rows * self.dim * 4)

    @property
    def allocated_bytes(self) -> int:
        return int(self._buffer.nbytes)

    def snapshot(self) -> dict[str, int | float]:
        return {"rows": self._rows, "capacity": self.capacity,
                "appends": self.appends, "reallocations": self.reallocations,
                "used_bytes": self.nbytes, "allocated_bytes": self.allocated_bytes,
                "load_factor": round(self._rows / self.capacity, 3) if self.capacity else 0.0}
