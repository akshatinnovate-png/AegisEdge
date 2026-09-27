"""Projecting the stored vectors down to something a person can look at.

A memory lives in 256 dimensions. A screen has two. Any picture of a vector
space is therefore a lie of compression, and the only question is whether it
says how much it left out.

This is a plain PCA — centre, SVD, keep two components — and it returns the
share of variance those two components carry alongside the coordinates. On a
well-spread corpus that share is small, which is the honest headline: two
points drawn next to each other may be nothing of the sort, and the number
under the plot is what stops the picture from being read as a measurement.

The query is projected through the *same* basis rather than being fitted with
the documents. Refitting to include it would move every document a little to
accommodate one vector, and the plot is meant to show where the query landed
in the corpus's own space, not in a space bent around it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Projection:
    coords: np.ndarray                              # (n, 2)
    explained: float = 0.0                          # share of variance in 2D
    centre: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    basis: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))

    def project(self, vectors: np.ndarray) -> np.ndarray:
        """Put new vectors into the basis the corpus defined."""
        array = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
        if not self.basis.size or array.shape[1] != self.centre.shape[0]:
            return np.zeros((array.shape[0], 2), dtype=np.float32)
        return (array - self.centre) @ self.basis


def project(vectors: np.ndarray) -> Projection:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0:
        return Projection(np.zeros((0, 2), dtype=np.float32))
    if array.shape[0] == 1:
        # One point is at the origin of its own space, and its "explained
        # variance" is not 1.0 — there is no variance to explain.
        return Projection(np.zeros((1, 2), dtype=np.float32), 0.0,
                          array[0].copy(), np.zeros((array.shape[1], 2), dtype=np.float32))
    centre = array.mean(axis=0)
    centred = array - centre
    _, singular, right = np.linalg.svd(centred, full_matrices=False)
    components = min(2, right.shape[0])
    basis = np.ascontiguousarray(right[:components].T)
    if components == 1:                              # a one-dimensional corpus
        basis = np.hstack([basis, np.zeros((basis.shape[0], 1), dtype=np.float32)])
    energy = float((singular ** 2).sum())
    explained = float((singular[:components] ** 2).sum() / energy) if energy else 0.0
    return Projection(centred @ basis, round(explained, 4), centre, basis.astype(np.float32))


def scale(coords: np.ndarray, extra: np.ndarray | None = None) -> dict[str, Any]:
    """Normalise to a 0..1 box so the client does not have to guess the range.

    The query is included in the extent when there is one, so a query that
    landed outside the corpus is drawn outside it rather than clamped onto the
    edge and made to look like a neighbour of whatever is there.
    """
    if coords.shape[0] == 0:
        return {"points": [], "query": None, "extent": None}
    stack = coords if extra is None or not len(extra) else np.vstack([coords, extra])
    low, high = stack.min(axis=0), stack.max(axis=0)
    span = np.where((high - low) == 0, 1.0, high - low)

    def unit(array: np.ndarray) -> list[list[float]]:
        return [[round(float(x), 5), round(float(y), 5)] for x, y in (array - low) / span]

    return {
        "points": unit(coords),
        "query": unit(np.atleast_2d(extra))[0] if extra is not None and len(extra) else None,
        "extent": [[float(low[0]), float(low[1])], [float(high[0]), float(high[1])]],
    }
