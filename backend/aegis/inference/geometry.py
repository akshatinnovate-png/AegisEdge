"""Corpus-adapted embedding geometry, fitted on the device.

Pretrained embedding spaces are badly conditioned for cosine retrieval. The
measurement is easy to make and hard to argue with: take two unrelated
sentences, embed them, and the cosine is not near 0 — it is typically 0.4-0.8.
Every vector sits inside a narrow cone, a handful of principal directions carry
most of the variance, and the similarity you actually wanted is a small
perturbation on top of a large common component. Mu & Viswanath (ICLR 2018)
named the fix "all-but-the-top"; Su et al. (2021) showed plain whitening
recovers most of the same gain with no training.

Both need one thing this device has and a model vendor does not: the
covariance of *this corpus*. A packing line and a clinic have different
principal directions, so a transform shipped in the wheel would be fitted to
neither. Here it is estimated from the vectors the node has actually stored,
refitted as the corpus drifts, versioned, and migrated through the same
dual-space machinery that handles a model change — because it *is* a model
change, and pretending otherwise would silently corrupt every stored vector.

What it produces:

- a whitening transform that makes the space isotropic, so cosine means what
  cosine is supposed to mean;
- a principal-direction ordering, which makes dimension truncation principled
  rather than arbitrary — a 64-dimensional sketch that keeps the top-variance
  directions is a usable first-pass filter, where the first 64 raw dimensions
  of a pretrained table are not;
- the diagnostics that justify both, so the claim is measured and not asserted.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np


class AnisotropyProbe:
    """Measure how badly a set of embeddings is conditioned for cosine."""

    @staticmethod
    def measure(vectors: np.ndarray, sample: int = 2048,
                seed: int = 0) -> dict[str, Any]:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] < 4:
            return {"measurable": False, "reason": "need at least 4 vectors"}

        rng = np.random.default_rng(seed)
        if matrix.shape[0] > sample:
            matrix = matrix[rng.choice(matrix.shape[0], sample, replace=False)]
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit = matrix / norms

        # Mean cosine between *unrelated* pairs. In a well-conditioned space
        # this is ~0; the distance from 0 is the size of the common component
        # every similarity score is paying for.
        n = unit.shape[0]
        left = rng.integers(0, n, size=min(4096, n * 4))
        right = rng.integers(0, n, size=left.size)
        keep = left != right
        pairs = float(np.mean(np.einsum("ij,ij->i", unit[left[keep]], unit[right[keep]])))

        centred = unit - unit.mean(axis=0, keepdims=True)
        # Economy SVD on the sample: singular values squared are the spectrum.
        singular = np.linalg.svd(centred, compute_uv=False)
        spectrum = (singular ** 2) / max(centred.shape[0] - 1, 1)
        total = float(spectrum.sum()) or 1.0
        share = spectrum / total

        # Effective rank (Roy & Vetterli): exp of the spectral entropy. A
        # 256-dimensional space with an effective rank of 30 is a
        # 30-dimensional space wearing a costume.
        nonzero = share[share > 1e-12]
        entropy = float(-(nonzero * np.log(nonzero)).sum())
        effective_rank = float(np.exp(entropy))

        return {
            "measurable": True,
            "sampled": int(unit.shape[0]),
            "dim": int(unit.shape[1]),
            "mean_random_pair_cosine": round(pairs, 4),
            "top1_variance_share": round(float(share[0]), 4),
            "top10_variance_share": round(float(share[:10].sum()), 4),
            "effective_rank": round(effective_rank, 1),
            "effective_rank_ratio": round(effective_rank / unit.shape[1], 4),
            "condition_number": round(float(spectrum[0] / max(spectrum[-1], 1e-12)), 1),
            "verdict": AnisotropyProbe._verdict(pairs, effective_rank / unit.shape[1]),
        }

    @staticmethod
    def _verdict(pair_cosine: float, rank_ratio: float) -> str:
        if pair_cosine < 0.1 and rank_ratio > 0.5:
            return "isotropic — whitening has little left to recover"
        if pair_cosine > 0.4 or rank_ratio < 0.2:
            return "strongly anisotropic — cosine is dominated by a common component"
        return "moderately anisotropic — whitening should measurably help"


def ledoit_wolf_shrinkage(sample: np.ndarray, covariance: np.ndarray,
                          n_total: int | None = None) -> float:
    """The optimal shrinkage intensity towards a scaled identity.

    A covariance estimated from n vectors in d dimensions is badly conditioned
    whenever n is not enormously larger than d, and inverting a badly
    conditioned covariance — which is what whitening does — amplifies exactly
    the directions the estimate is worst at. Ledoit & Wolf (2004) give the
    shrinkage weight that minimises expected squared error in closed form, so
    the node does not have to guess a ridge constant.

    ``sample`` may be a uniform subsample used only to estimate the fourth
    moment; ``n_total`` is then the real number of vectors the covariance was
    built from. Passing the subsample's own size here is wrong and drives the
    intensity to 1, which silently replaces whitening with a bare rotation.
    """
    m, d = sample.shape
    n = int(n_total) if n_total else m
    if m < 2 or n < 2:
        return 1.0
    mu = float(np.trace(covariance) / d)
    d2 = float(((covariance - mu * np.eye(d, dtype=covariance.dtype)) ** 2).sum())
    if d2 <= 0:
        return 1.0
    # b^2 = (1/n^2) sum_i || x_i x_i^T - S ||_F^2. Expanding and using
    # sum_i x_i^T S x_i = n * ||S||_F^2 collapses it to a fourth moment, so no
    # d x d matrix is ever materialised per sample.
    frobenius = float((covariance ** 2).sum())
    fourth = float((np.einsum("ij,ij->i", sample, sample) ** 2).mean())
    b2 = max((fourth - frobenius) / n, 0.0)
    b2 = min(b2, d2)
    return float(min(max(b2 / d2, 0.0), 1.0))


class GeometryVersion:
    """Identity of a fitted transform. A vector is only comparable within one."""

    __slots__ = ("id", "fitted_at", "samples", "in_dim", "out_dim", "dropped",
                 "shrinkage", "energy_retained")

    def __init__(self, digest: str, fitted_at: float, samples: int, in_dim: int,
                 out_dim: int, dropped: int, shrinkage: float,
                 energy_retained: float) -> None:
        self.id = digest
        self.fitted_at = fitted_at
        self.samples = samples
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropped = dropped
        self.shrinkage = shrinkage
        self.energy_retained = energy_retained

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "fitted_at": self.fitted_at, "samples": self.samples,
                "in_dim": self.in_dim, "out_dim": self.out_dim,
                "dropped_components": self.dropped,
                "shrinkage": round(self.shrinkage, 4),
                "energy_retained": round(self.energy_retained, 4)}


class CorpusGeometry:
    """Streaming covariance, whitening transform, principled truncation.

    Statistics accumulate continuously and cheaply (one BLAS rank-k update per
    batch). Fitting — the eigendecomposition — happens only when asked, on the
    maintenance lane, because it is O(d^3) and has no business on a write path.
    """

    __slots__ = ("dim", "drop_components", "reservoir_size", "energy_target",
                 "min_eigenvalue_ratio", "_n", "_sum", "_gram",
                 "_reservoir", "_seen", "_rng", "_lock", "mean", "transform_matrix",
                 "spectrum", "version", "fits", "_enabled", "rank")

    def __init__(self, dim: int, drop_components: int | None = None,
                 reservoir_size: int = 4096, seed: int = 0,
                 energy_target: float = 0.95,
                 min_eigenvalue_ratio: float = 1e-4) -> None:
        self.dim = int(dim)
        # Mu & Viswanath remove the top d/100 directions. Measured on this
        # encoder it is the wrong move: `scripts/geometry_eval.py` puts
        # drop=0 at R@1 0.488 and drop=1 at 0.476, and dropping even one
        # component collapses cold-tier 1-bit recall from 0.89 to 0.16,
        # because what is left is dominated by low-variance directions a sign
        # bit cannot resolve. Centring plus rank-limited whitening already
        # removes the common component the paper was targeting. Default 0,
        # and the knob stays for a corpus that measures differently.
        self.drop_components = 0 if drop_components is None else int(drop_components)
        self.reservoir_size = int(reservoir_size)
        # Whitening divides by sqrt(eigenvalue). In a space whose effective
        # rank is a tenth of its dimension — which is what a pretrained table
        # actually looks like — most eigenvalues are numerically zero, and
        # dividing by their square roots amplifies pure noise by four orders of
        # magnitude. Measured on the real encoder, unrestricted whitening took
        # cold-tier recall@10 from 0.77 to 0.11. So the transform keeps a rank,
        # not a dimension: enough directions to hold `energy_target` of the
        # variance, and never one whose eigenvalue has fallen through the
        # conditioning floor.
        self.energy_target = float(energy_target)
        self.min_eigenvalue_ratio = float(min_eigenvalue_ratio)
        self.rank = 0
        self._n = 0
        self._sum = np.zeros(self.dim, dtype=np.float64)
        self._gram = np.zeros((self.dim, self.dim), dtype=np.float64)
        self._reservoir = np.zeros((self.reservoir_size, self.dim), dtype=np.float32)
        self._seen = 0
        self._rng = np.random.default_rng(seed)
        self._lock = threading.Lock()

        self.mean: np.ndarray | None = None
        self.transform_matrix: np.ndarray | None = None
        self.spectrum: np.ndarray | None = None
        self.version: GeometryVersion | None = None
        self.fits = 0
        self._enabled = False

    # -- accumulate -------------------------------------------------------

    def observe(self, vectors: np.ndarray) -> int:
        """Fold vectors into the running moments. O(n·d²) in BLAS, no copies kept."""
        matrix = np.asarray(vectors, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.size == 0 or matrix.shape[1] != self.dim:
            return 0
        with self._lock:
            self._n += matrix.shape[0]
            self._sum += matrix.sum(axis=0)
            self._gram += matrix.T @ matrix
            self._fill_reservoir(matrix)
        return int(matrix.shape[0])

    def _fill_reservoir(self, matrix: np.ndarray) -> None:
        """Algorithm R: a uniform sample of the stream, for the shrinkage estimate."""
        for row in matrix:
            if self._seen < self.reservoir_size:
                self._reservoir[self._seen] = row
            else:
                slot = int(self._rng.integers(0, self._seen + 1))
                if slot < self.reservoir_size:
                    self._reservoir[slot] = row
            self._seen += 1

    # -- fit --------------------------------------------------------------

    def fit(self, min_samples: int | None = None) -> GeometryVersion | None:
        """Estimate the transform. Refuses rather than fits noise."""
        floor = min_samples if min_samples is not None else max(4 * self.dim, 512)
        with self._lock:
            if self._n < floor:
                return None
            n = self._n
            mean = self._sum / n
            # Covariance of the centred data, from the uncentred Gram matrix.
            covariance = (self._gram / n) - np.outer(mean, mean)
            sample = self._reservoir[: min(self._seen, self.reservoir_size)].astype(np.float64)
            sample = sample - mean

        covariance = 0.5 * (covariance + covariance.T)      # kill float asymmetry
        shrinkage = (ledoit_wolf_shrinkage(sample, covariance, n_total=n)
                     if sample.shape[0] > 1 else 1.0)
        target = float(np.trace(covariance) / self.dim)
        regularised = ((1.0 - shrinkage) * covariance
                       + shrinkage * target * np.eye(self.dim))

        eigenvalues, eigenvectors = np.linalg.eigh(regularised)
        order = np.argsort(-eigenvalues)                    # descending variance
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        drop = min(self.drop_components, self.dim - 1)
        available = eigenvalues[drop:]
        total_energy = float(eigenvalues.sum()) or 1.0
        rank = self._choose_rank(eigenvalues, available, total_energy)
        kept_values = available[:rank]
        kept_vectors = eigenvectors[:, drop:drop + rank]
        retained = float(kept_values.sum()) / total_energy

        # W = V · Λ^{-1/2}: project onto the principal directions, then equalise
        # their scales. Columns stay ordered by the variance they *had*, which
        # is what makes truncating to the first m columns meaningful.
        floor_value = max(float(kept_values.max()) * 1e-8, 1e-12)
        scales = 1.0 / np.sqrt(np.maximum(kept_values, floor_value))
        matrix = (kept_vectors * scales).astype(np.float32)

        digest = hashlib.sha256(
            matrix.tobytes() + mean.astype(np.float32).tobytes()).hexdigest()[:16]

        with self._lock:
            self.mean = mean.astype(np.float32)
            self.transform_matrix = matrix
            self.spectrum = eigenvalues.astype(np.float32)
            self.rank = int(rank)
            self.fits += 1
            self.version = GeometryVersion(digest, time.time(), n, self.dim,
                                           int(matrix.shape[1]), drop, shrinkage, retained)
        return self.version

    def _choose_rank(self, spectrum: np.ndarray, available: np.ndarray,
                     total: float) -> int:
        """How many directions carry signal rather than rounding error.

        Two independent brakes, and the tighter one wins:

        * **Energy.** Keep the fewest directions holding `energy_target` of the
          total variance. This is the rank the data says it has.
        * **Conditioning.** Never keep a direction whose eigenvalue has fallen
          below `min_eigenvalue_ratio` of the largest. Whitening multiplies by
          the inverse square root, so a ratio of 1e-8 is a gain of 1e4 applied
          to a direction that is numerically indistinguishable from zero.
        """
        if available.size == 0:
            return 1
        cumulative = np.cumsum(available) / max(total, 1e-12)
        by_energy = int(np.searchsorted(cumulative, self.energy_target) + 1)
        floor = float(available[0]) * self.min_eigenvalue_ratio
        by_conditioning = int(np.count_nonzero(available >= floor))
        return max(1, min(available.size, by_energy, max(by_conditioning, 1)))

    # -- apply ------------------------------------------------------------

    @property
    def fitted(self) -> bool:
        return self.transform_matrix is not None

    @property
    def enabled(self) -> bool:
        return self._enabled and self.fitted

    def enable(self, on: bool = True) -> bool:
        """Arm the transform. Off until something has migrated the corpus."""
        self._enabled = bool(on) and self.fitted
        return self._enabled

    @property
    def out_dim(self) -> int:
        if self.transform_matrix is None:
            return self.dim
        return int(self.transform_matrix.shape[1])

    def transform(self, vectors: np.ndarray, dims: int | None = None) -> np.ndarray:
        """Whiten, optionally truncate to the top `dims` directions, re-normalise."""
        matrix = np.asarray(vectors, dtype=np.float32)
        single = matrix.ndim == 1
        if single:
            matrix = matrix.reshape(1, -1)
        if self.transform_matrix is None or self.mean is None:
            out = matrix
        else:
            projection = self.transform_matrix
            if dims is not None:
                projection = projection[:, : max(1, min(int(dims), projection.shape[1]))]
            out = (matrix - self.mean) @ projection
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        out = (out / norms).astype(np.float32, copy=False)
        return out[0] if single else out

    def energy_retained(self, dims: int) -> float:
        """Share of the original variance the first `dims` output directions carry."""
        if self.spectrum is None:
            return 0.0
        drop = min(self.drop_components, self.dim - 1)
        kept = self.spectrum[drop:drop + max(self.rank, 1)]
        total = float(self.spectrum.sum()) or 1.0
        return float(kept[: max(1, min(dims, kept.size))].sum()) / total

    def nested_ladder(self, steps: tuple[int, ...] = (8, 16, 32, 64, 128, 192, 256)) -> list[dict]:
        """A Matryoshka-style dimension ladder, with what each rung actually keeps.

        Not trained Matryoshka representation learning — this is the PCA
        ordering, which is the honest version available without retraining the
        encoder: the directions are ranked by variance, so a prefix is the
        best rank-m approximation of the space in squared error.
        """
        if self.spectrum is None:
            return []
        return [{"dims": d, "energy_retained": round(self.energy_retained(d), 4),
                 "bytes_per_vector": d * 4, "compression": round(self.dim / d, 2)}
                for d in steps if d <= self.out_dim]

    # -- persistence ------------------------------------------------------

    def save(self, path: Path) -> bool:
        if self.transform_matrix is None or self.mean is None or self.version is None:
            return False
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        # Hand savez a file object: given a *path* it silently appends ".npz",
        # and the atomic rename then targets a file that was never written.
        with open(temp, "wb") as handle:
            np.savez_compressed(
                handle, mean=self.mean, matrix=self.transform_matrix,
                spectrum=self.spectrum if self.spectrum is not None else np.zeros(0),
                meta=np.frombuffer(json.dumps(
                    {**self.version.as_dict(), "enabled": self._enabled,
                     "rank": self.rank,
                     "drop_components": self.drop_components}).encode(), dtype=np.uint8))
        temp.replace(path)
        return True

    def load(self, path: Path) -> GeometryVersion | None:
        path = Path(path)
        if not path.exists():
            return None
        try:
            with np.load(path) as bundle:
                meta = json.loads(bytes(bundle["meta"]).decode())
                if int(meta.get("in_dim", -1)) != self.dim:
                    return None            # a transform for a different space
                self.mean = bundle["mean"].astype(np.float32)
                self.transform_matrix = bundle["matrix"].astype(np.float32)
                spectrum = bundle["spectrum"]
                self.spectrum = spectrum.astype(np.float32) if spectrum.size else None
                self.drop_components = int(meta.get("drop_components", self.drop_components))
                self.rank = int(meta.get("rank", self.transform_matrix.shape[1]))
                self.version = GeometryVersion(
                    meta["id"], meta["fitted_at"], meta["samples"], meta["in_dim"],
                    meta["out_dim"], meta["dropped_components"], meta["shrinkage"],
                    meta["energy_retained"])
                self._enabled = bool(meta.get("enabled", False))
            return self.version
        except Exception:
            return None

    def snapshot(self) -> dict[str, Any]:
        return {
            "observed": self._n,
            "reservoir": int(min(self._seen, self.reservoir_size)),
            "fits": self.fits,
            "fitted": self.fitted,
            "enabled": self.enabled,
            "in_dim": self.dim,
            "out_dim": self.out_dim,
            "drop_components": self.drop_components,
            "rank": self.rank,
            "effective_rank_used": self.rank,
            "energy_target": self.energy_target,
            "version": self.version.as_dict() if self.version else None,
            "nested_ladder": self.nested_ladder(),
        }
