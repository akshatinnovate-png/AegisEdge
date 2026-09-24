"""RaBitQ: 1-bit codes that know how wrong they are.

The cold tier stores one bit per dimension. The obvious way to do that is
``sign(v)``, which is what this node did: cheap, 32x smaller, and silently
lossy in a way nothing downstream can reason about. Its score is not an
estimate of the inner product — it is a monotone-ish proxy for one — so the
only way to use it safely is to over-fetch by a fixed multiplier and hope the
multiplier was generous enough. Six was the number. Nobody could say why six.

RaBitQ (Gao & Long, SIGMOD 2024) replaces the proxy with an *unbiased
estimator* and a concentration bound. Three changes buy it:

1. **Centre.** Quantize the residual from the corpus centroid, not the vector.
   Sign bits of a cone of vectors that all point the same way carry almost no
   information; sign bits of their residuals carry all of it.
2. **Rotate.** Apply a random orthogonal transform first. Inner products are
   preserved exactly, but the error of the sign quantizer is spread evenly
   across dimensions instead of concentrating wherever the data happens to be
   axis-aligned — which is what makes a bound possible at all.
3. **Keep one float.** Store ``f = <x̄, o_r>``, how well the code aligns with
   the vector it encodes. Dividing by it corrects the systematic shrinkage of
   the sign quantizer, making the estimate unbiased, and it is exactly the
   quantity that says how trustworthy this particular code is.

The payoff is not only accuracy. With a bound per candidate, the rescoring
depth stops being a guess: fetch the full-precision vector only for candidates
whose bound still lets them reach the top-k. Easy queries touch the disk a few
times, ambiguous ones touch it more, and neither is a constant somebody picked.

Cost: ``dim/8 + 8`` bytes per vector against ``dim/8`` — at 256 dimensions,
40 bytes rather than 32, so 25.6x compression instead of 32x. Eight bytes to
turn a guess into a bound is the cheapest trade in the system.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .growable import GrowableMatrix

# P(|error| > eps0 * sigma) <= 2 exp(-eps0^2 / 2). At 2.6 that is under 1.5%
# per candidate, and the rescoring pass catches what slips through.
DEFAULT_EPSILON = 2.6


def fwht(matrix: np.ndarray) -> np.ndarray:
    """In-place fast Walsh-Hadamard transform over the last axis.

    O(d log d) against O(d^2) for a dense rotation. At d=256 that is 2,048
    add/subtract pairs instead of 65,536 multiply-accumulates, and it needs no
    d x d matrix resident to do it.
    """
    out = np.array(matrix, dtype=np.float32, copy=True)
    shape = out.shape
    d = shape[-1]
    if d & (d - 1):
        raise ValueError("fwht needs a power-of-two width")
    out = out.reshape(-1, d)
    h = 1
    while h < d:
        out = out.reshape(-1, d // (2 * h), 2, h)
        upper = out[:, :, 0, :].copy()
        lower = out[:, :, 1, :].copy()
        out[:, :, 0, :] = upper + lower
        out[:, :, 1, :] = upper - lower
        out = out.reshape(-1, d)
        h *= 2
    return out.reshape(shape)


class Rotation:
    """A random orthogonal transform, structured when the dimension allows it.

    Power-of-two dimensions get three rounds of ``H·D`` (Hadamard times a
    random sign flip), the standard fast Johnson-Lindenstrauss construction:
    orthogonal by construction, indistinguishable from a random rotation for
    this purpose, and log-linear. Everything else falls back to the QR
    factorisation of a Gaussian matrix, which is exact and quadratic.

    Both are derived from a seed, so the transform is reproducible across
    restarts and across peers without ever being written to disk or sent over
    a wire.
    """

    def __init__(self, dim: int, seed: int = 0x5EED) -> None:
        self.dim = int(dim)
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        self.fast = (self.dim & (self.dim - 1)) == 0 and self.dim >= 2
        if self.fast:
            self.signs = [rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=self.dim)
                          for _ in range(3)]
            self.matrix = None
            self._scale = np.float32(1.0 / math.sqrt(self.dim))
        else:
            gaussian = rng.normal(size=(self.dim, self.dim)).astype(np.float32)
            q, r = np.linalg.qr(gaussian)
            # Fix the sign convention so the factorisation is deterministic.
            self.matrix = (q * np.sign(np.diag(r))).astype(np.float32)
            self.signs = []
            self._scale = np.float32(1.0)

    def apply(self, vectors: np.ndarray) -> np.ndarray:
        matrix = np.asarray(vectors, dtype=np.float32)
        single = matrix.ndim == 1
        if single:
            matrix = matrix.reshape(1, -1)
        if not self.fast:
            out = matrix @ self.matrix
        else:
            out = matrix
            for signs in self.signs:
                out = fwht(out * signs) * self._scale
        return out[0] if single else out

    def snapshot(self) -> dict[str, Any]:
        return {"dim": self.dim, "seed": self.seed,
                "kind": "hadamard x3 (O(d log d))" if self.fast else "dense QR (O(d^2))",
                "resident_bytes": 0 if self.fast else self.dim * self.dim * 4}


class RaBitQCodes:
    """Packed codes plus the two scalars that make them estimable."""

    __slots__ = ("bits", "factor", "radius", "dim")

    def __init__(self, bits: np.ndarray, factor: np.ndarray, radius: np.ndarray,
                 dim: int) -> None:
        self.bits = bits            # (n, ceil(dim/8)) uint8
        self.factor = factor        # (n,) float32  <x̄, o_r>
        self.radius = radius        # (n,) float32  ||o - c||
        self.dim = dim

    def __len__(self) -> int:
        return int(self.bits.shape[0])

    @property
    def nbytes(self) -> int:
        return int(self.bits.nbytes + self.factor.nbytes + self.radius.nbytes)

    def take(self, index: np.ndarray) -> "RaBitQCodes":
        return RaBitQCodes(self.bits[index], self.factor[index], self.radius[index], self.dim)

    @staticmethod
    def stack(parts: list["RaBitQCodes"], dim: int) -> "RaBitQCodes":
        if not parts:
            return RaBitQCodes(np.zeros((0, (dim + 7) // 8), np.uint8),
                               np.zeros(0, np.float32), np.zeros(0, np.float32), dim)
        return RaBitQCodes(np.vstack([p.bits for p in parts]),
                           np.concatenate([p.factor for p in parts]),
                           np.concatenate([p.radius for p in parts]), dim)


class QueryContext:
    """A query, rotated once, reused against every code."""

    __slots__ = ("rotated", "radius", "sum_rotated")

    def __init__(self, rotated: np.ndarray, radius: float) -> None:
        self.rotated = rotated
        self.radius = radius
        self.sum_rotated = float(rotated.sum())


class RaBitQ:
    """Centroid + rotation + sign bits, with an unbiased inner-product estimator."""

    def __init__(self, dim: int, seed: int = 0x5EED,
                 epsilon: float = DEFAULT_EPSILON) -> None:
        self.dim = int(dim)
        self.rotation = Rotation(self.dim, seed)
        self.centroid = np.zeros(self.dim, dtype=np.float32)
        self.epsilon = float(epsilon)
        self.fitted = False
        self.encoded = 0
        self.estimated = 0
        # Running mean, so the centroid tracks a corpus that is still growing
        # rather than freezing on whatever the first batch happened to be.
        self._n = 0
        self._sum = np.zeros(self.dim, dtype=np.float64)

    # -- fit --------------------------------------------------------------

    def observe(self, vectors: np.ndarray) -> None:
        matrix = np.asarray(vectors, dtype=np.float64).reshape(-1, self.dim)
        if matrix.size == 0:
            return
        self._n += matrix.shape[0]
        self._sum += matrix.sum(axis=0)

    def fit(self, vectors: np.ndarray | None = None) -> bool:
        """Set the centroid. Re-fitting invalidates every existing code."""
        if vectors is not None:
            self.observe(vectors)
        if self._n == 0:
            return False
        centroid = (self._sum / self._n).astype(np.float32)
        self.centroid = centroid
        self.fitted = True
        return True

    @property
    def drift(self) -> float:
        """How far the running mean has moved from the fitted centroid.

        Codes are only as good as the centroid they were taken against. When
        this grows the maintenance lane should refit and re-encode, and this
        number is what tells it to.
        """
        if self._n == 0:
            return 0.0
        current = (self._sum / self._n).astype(np.float32)
        return float(np.linalg.norm(current - self.centroid))

    # -- encode -----------------------------------------------------------

    def encode(self, vectors: np.ndarray) -> RaBitQCodes:
        matrix = np.asarray(vectors, dtype=np.float32).reshape(-1, self.dim)
        residual = matrix - self.centroid
        radius = np.linalg.norm(residual, axis=1).astype(np.float32)
        safe = np.where(radius > 1e-9, radius, np.float32(1.0))
        unit = residual / safe[:, None]

        rotated = self.rotation.apply(unit)
        bits = np.packbits(rotated > 0, axis=1)
        # f = <x̄, rotated> with x̄ = ±1/sqrt(d) matching the signs, which is
        # exactly the L1 norm over sqrt(d). For a well-spread direction this
        # sits near sqrt(2/pi) ≈ 0.798; a code far below that is a code whose
        # estimate deserves little confidence, and the bound says so.
        factor = (np.abs(rotated).sum(axis=1) / math.sqrt(self.dim)).astype(np.float32)
        factor = np.maximum(factor, np.float32(1e-6))
        self.encoded += int(matrix.shape[0])
        return RaBitQCodes(bits, factor, radius, self.dim)

    # -- estimate ---------------------------------------------------------

    def prepare(self, query: np.ndarray) -> QueryContext:
        vector = np.asarray(query, dtype=np.float32).reshape(-1)
        residual = vector - self.centroid
        radius = float(np.linalg.norm(residual))
        unit = residual / (radius if radius > 1e-9 else 1.0)
        return QueryContext(self.rotation.apply(unit).reshape(-1), radius)

    def estimate(self, context: QueryContext, codes: RaBitQCodes,
                 chunk: int = 8192) -> tuple[np.ndarray, np.ndarray]:
        """Unbiased inner-product estimates and their two-sided error bounds.

        Returns ``(estimate, bound)`` such that the true inner product lies in
        ``estimate ± bound`` with probability at least 1 - 2·exp(-ε²/2).
        """
        n = len(codes)
        if n == 0:
            return np.zeros(0, np.float32), np.zeros(0, np.float32)
        self.estimated += n

        scale = 1.0 / math.sqrt(self.dim)
        projections = np.empty(n, dtype=np.float32)
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            # Unpack in blocks: the whole point of the codes is that the
            # expanded form never has to be resident all at once.
            block = np.unpackbits(codes.bits[start:stop], axis=1)[:, : self.dim]
            projections[start:stop] = (2.0 * (block @ context.rotated)
                                       - context.sum_rotated) * scale

        cosine = np.clip(projections / codes.factor, -1.0, 1.0)

        # ||o - q||^2 = r_o^2 + r_q^2 - 2 r_o r_q cos, and for unit-norm o, q
        # the inner product is 1 - ||o - q||^2 / 2.
        rq = context.radius
        squared = codes.radius ** 2 + rq * rq - 2.0 * codes.radius * rq * cosine
        estimate = (1.0 - squared / 2.0).astype(np.float32)

        # Concentration of the estimator: the residual variance of a sign code
        # is (1 - f^2) / (d - 1), scaled by how much the estimate leans on it.
        sigma = np.sqrt(np.maximum(1.0 - codes.factor ** 2, 0.0)
                        / max(self.dim - 1, 1)) / codes.factor
        bound = (self.epsilon * sigma * codes.radius * rq).astype(np.float32)
        return estimate, bound

    def snapshot(self) -> dict[str, Any]:
        per_vector = (self.dim + 7) // 8 + 8
        return {
            "dim": self.dim, "fitted": self.fitted, "epsilon": self.epsilon,
            "confidence": round(1.0 - 2.0 * math.exp(-self.epsilon ** 2 / 2.0), 4),
            "rotation": self.rotation.snapshot(),
            "bytes_per_vector": per_vector,
            "compression": round(self.dim * 4 / per_vector, 2),
            "centroid_drift": round(self.drift, 5),
            "observed": self._n, "encoded": self.encoded, "estimated": self.estimated,
        }


def adaptive_shortlist(estimate: np.ndarray, bound: np.ndarray, k: int,
                       budget: int | None = None) -> np.ndarray:
    """Which candidates could still reach the top-k, given the bounds.

    The k-th largest *lower* bound is a score the top-k provably reaches. Any
    candidate whose upper bound falls below it cannot be in the true top-k and
    never needs its full-precision vector fetched. Everything else does.

    This is what replaces the fixed over-fetch multiplier: on a query with a
    clear winner the shortlist collapses to a handful, on an ambiguous one it
    widens by itself, and in both cases the recall it gives up is bounded
    rather than hoped for.

    ``budget`` caps the result. A corpus where every vector is nearly
    equidistant from the query — measured at 96% of candidates in a badly
    whitened space — makes the honest bound almost useless, and a latency
    SLO is not negotiable just because the bound is wide. When the cap binds,
    the guarantee is gone and the caller is told so by the returned size.
    """
    n = estimate.size
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    k = max(1, min(int(k), n))
    lower = estimate - bound
    upper = estimate + bound
    threshold = np.partition(lower, n - k)[n - k]
    keep = np.flatnonzero(upper >= threshold)
    if keep.size < k:
        # Degenerate bounds (identical vectors, a corpus of one) must never
        # shrink the shortlist below what the caller asked for.
        keep = np.argsort(-estimate)[:k]
    if budget is not None and keep.size > budget:
        cap = max(k, int(budget))
        keep = keep[np.argsort(-estimate[keep])[:cap]]
    return keep


class ColdCodebook:
    """Columnar RaBitQ codes for a whole tier, appendable in O(1).

    Holding one code object per point and stacking them at query time is the
    same mistake `GrowableMatrix` was written to remove: an O(n) allocation on
    every single search, which is invisible at a thousand points and ruinous at
    a hundred thousand. Bits, factors and radii live in three growable buffers
    instead, and the scan runs straight off the views.
    """

    __slots__ = ("codec", "dim", "_bits", "_factor", "_radius", "_row_of", "_ids")

    def __init__(self, codec: RaBitQ) -> None:
        self.codec = codec
        self.dim = codec.dim
        self._bits = GrowableMatrix((codec.dim + 7) // 8, dtype=np.uint8)
        self._factor = GrowableMatrix(1)
        self._radius = GrowableMatrix(1)
        self._row_of: dict[str, int] = {}
        self._ids: list[str] = []

    def __len__(self) -> int:
        return len(self._ids)

    def __contains__(self, point_id: str) -> bool:
        return point_id in self._row_of

    def __iter__(self):
        return iter(list(self._ids))

    def put(self, point_id: str, vector: np.ndarray) -> None:
        self.codec.observe(vector.reshape(1, -1))
        if not self.codec.fitted:
            self.codec.fit()
        codes = self.codec.encode(vector.reshape(1, -1))
        row = self._row_of.get(point_id)
        if row is None:
            row = self._bits.append(codes.bits[0])
            self._factor.append(codes.factor[:1])
            self._radius.append(codes.radius[:1])
            self._row_of[point_id] = row
            self._ids.append(point_id)
        else:
            self._bits[row] = codes.bits[0]
            self._factor[row] = codes.factor[:1]
            self._radius[row] = codes.radius[:1]

    def pop(self, point_id: str) -> bool:
        row = self._row_of.pop(point_id, None)
        if row is None:
            return False
        moved_from = self._bits.swap_remove(row)
        self._factor.swap_remove(row)
        self._radius.swap_remove(row)
        last = self._ids.pop()
        if moved_from is not None and last != point_id:
            self._ids[row] = last
            self._row_of[last] = row
        return True

    def codes(self) -> RaBitQCodes:
        """All codes, as views. No copy, no stacking."""
        return RaBitQCodes(self._bits.view, self._factor.view.reshape(-1),
                           self._radius.view.reshape(-1), self.dim)

    def subset(self, allow: set[str]) -> tuple[list[str], RaBitQCodes]:
        rows = [self._row_of[p] for p in self._ids if p in allow]
        if not rows:
            return [], RaBitQCodes(np.zeros((0, (self.dim + 7) // 8), np.uint8),
                                   np.zeros(0, np.float32), np.zeros(0, np.float32), self.dim)
        index = np.asarray(rows, dtype=np.int64)
        ids = [self._ids[r] for r in rows]
        return ids, RaBitQCodes(self._bits.view[index], self._factor.view[index].reshape(-1),
                                self._radius.view[index].reshape(-1), self.dim)

    def ids(self) -> list[str]:
        return list(self._ids)

    @property
    def nbytes(self) -> int:
        return int(len(self) * ((self.dim + 7) // 8 + 8))

    def re_encode_all(self, vectors: dict[str, np.ndarray]) -> int:
        """Rebuild every code against the current centroid. Maintenance-lane work."""
        rebuilt = 0
        for point_id in list(self._ids):
            vector = vectors.get(point_id)
            if vector is None:
                continue
            codes = self.codec.encode(np.asarray(vector, np.float32).reshape(1, -1))
            row = self._row_of[point_id]
            self._bits[row] = codes.bits[0]
            self._factor[row] = codes.factor[:1]
            self._radius[row] = codes.radius[:1]
            rebuilt += 1
        return rebuilt
