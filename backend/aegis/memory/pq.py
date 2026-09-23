"""Product quantization and IVF-PQ.

Scalar int8 gets you 4x. Product quantization gets you 64x and more: the
vector is split into subspaces, each subspace gets its own 256-entry codebook
learned by k-means, and a point becomes one byte per subspace. A 384-dim
float32 vector (1536 B) becomes 48 B.

Search never decompresses. The query is projected once into a distance lookup
table — one row per subspace, 256 columns — and a candidate's distance is the
sum of eight table lookups (asymmetric distance computation). That is the
difference between scanning a cold tier and paging it in.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def kmeans(data: np.ndarray, k: int, iterations: int = 12, seed: int = 0) -> np.ndarray:
    """k-means++ init then Lloyd iterations. Small k, small data, no sklearn."""
    rng = np.random.default_rng(seed)
    n = len(data)
    if n <= k:
        pad = np.repeat(data[-1:], k - n, axis=0) if n < k else np.zeros((0, data.shape[1]), np.float32)
        return np.vstack([data, pad]).astype(np.float32)

    centroids = np.empty((k, data.shape[1]), dtype=np.float32)
    centroids[0] = data[rng.integers(n)]
    closest = ((data - centroids[0]) ** 2).sum(axis=1)
    for i in range(1, k):
        total = closest.sum()
        probabilities = closest / total if total > 0 else np.full(n, 1.0 / n)
        centroids[i] = data[rng.choice(n, p=probabilities)]
        closest = np.minimum(closest, ((data - centroids[i]) ** 2).sum(axis=1))

    for _ in range(iterations):
        assignments = np.argmin(
            ((data[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2), axis=1
        ) if n * k < 400_000 else _assign_chunked(data, centroids)
        moved = 0.0
        for cluster in range(k):
            members = data[assignments == cluster]
            if len(members):
                new = members.mean(axis=0)
                moved += float(np.abs(new - centroids[cluster]).sum())
                centroids[cluster] = new
        if moved < 1e-5:
            break
    return centroids


def _assign_chunked(data: np.ndarray, centroids: np.ndarray, chunk: int = 4096) -> np.ndarray:
    out = np.empty(len(data), dtype=np.int64)
    squared = (centroids ** 2).sum(axis=1)
    for start in range(0, len(data), chunk):
        block = data[start:start + chunk]
        distances = squared[None, :] - 2.0 * (block @ centroids.T)
        out[start:start + chunk] = np.argmin(distances, axis=1)
    return out


@dataclass
class PqParams:
    subspaces: int = 8              # m
    bits: int = 8                   # 2**bits centroids per subspace
    iterations: int = 12
    seed: int = 17
    opq_iterations: int = 6         # 0 disables the learned rotation

    @property
    def centroids(self) -> int:
        return 1 << self.bits


class ProductQuantizer:
    def __init__(self, dim: int, params: PqParams | None = None) -> None:
        self.dim = dim
        self.p = params or PqParams()
        if dim % self.p.subspaces:
            raise ValueError(f"dim {dim} must divide into {self.p.subspaces} subspaces")
        self.sub_dim = dim // self.p.subspaces
        self.codebooks: np.ndarray | None = None      # (m, 2**bits, sub_dim)
        self.rotation: np.ndarray | None = None       # OPQ: (dim, dim) orthonormal
        self.trained_on = 0
        self.residual_error = 0.0

    def _fit_codebooks(self, vectors: np.ndarray) -> np.ndarray:
        books = np.zeros((self.p.subspaces, self.p.centroids, self.sub_dim), dtype=np.float32)
        for sub in range(self.p.subspaces):
            block = vectors[:, sub * self.sub_dim:(sub + 1) * self.sub_dim]
            books[sub] = kmeans(block, self.p.centroids, self.p.iterations, self.p.seed + sub)
        return books

    def train(self, vectors: np.ndarray) -> "ProductQuantizer":
        """Train codebooks, optionally with an OPQ rotation.

        Plain PQ slices the vector by position, which assumes the dimensions
        are already grouped into independent blocks. Learned embeddings are
        nothing like that — variance is concentrated and correlated across
        arbitrary dimensions, so one subspace carries most of the error while
        others quantize near-constant values.

        OPQ learns an orthonormal rotation that spreads variance evenly before
        slicing, alternating between fitting codebooks and solving the
        orthogonal Procrustes problem for the rotation. A rotation is
        distance-preserving, so the query is simply rotated too.
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        self.rotation = np.eye(self.dim, dtype=np.float32)

        for _ in range(max(self.p.opq_iterations, 0)):
            rotated = vectors @ self.rotation
            self.codebooks = self._fit_codebooks(rotated)
            reconstructed = self.decode(self._encode_rotated(rotated))
            # min_R ||X R - X_hat||_F  subject to R orthonormal  →  R = U Vᵀ
            u, _, vt = np.linalg.svd(vectors.T @ reconstructed, full_matrices=False)
            self.rotation = (u @ vt).astype(np.float32)

        rotated = vectors @ self.rotation
        self.codebooks = self._fit_codebooks(rotated)
        self.trained_on = len(vectors)
        error = rotated - self.decode(self._encode_rotated(rotated))
        self.residual_error = float(np.mean(np.linalg.norm(error, axis=1)))
        return self

    def _encode_rotated(self, rotated: np.ndarray) -> np.ndarray:
        codes = np.empty((len(rotated), self.p.subspaces), dtype=np.uint8)
        for sub in range(self.p.subspaces):
            block = rotated[:, sub * self.sub_dim:(sub + 1) * self.sub_dim]
            book = self.codebooks[sub]
            distances = (block ** 2).sum(1)[:, None] - 2 * block @ book.T + (book ** 2).sum(1)[None, :]
            codes[:, sub] = np.argmin(distances, axis=1).astype(np.uint8)
        return codes

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        if self.codebooks is None:
            raise RuntimeError("quantizer is not trained")
        vectors = np.asarray(vectors, dtype=np.float32).reshape(-1, self.dim)
        return self._encode_rotated(vectors @ self.rotation)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        out = np.empty((len(codes), self.dim), dtype=np.float32)
        for sub in range(self.p.subspaces):
            out[:, sub * self.sub_dim:(sub + 1) * self.sub_dim] = self.codebooks[sub][codes[:, sub]]
        return out

    def lookup_table(self, query: np.ndarray) -> np.ndarray:
        """ADC table: inner product of each query sub-vector with each centroid.

        The query is rotated into the same learned basis; because the rotation
        is orthonormal, inner products are unchanged.
        """
        query = np.asarray(query, dtype=np.float32).reshape(-1) @ self.rotation
        table = np.empty((self.p.subspaces, self.p.centroids), dtype=np.float32)
        for sub in range(self.p.subspaces):
            block = query[sub * self.sub_dim:(sub + 1) * self.sub_dim]
            table[sub] = self.codebooks[sub] @ block
        return table

    @staticmethod
    def score(table: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """Sum of m table lookups per candidate — no decompression."""
        if not len(codes):
            return np.zeros(0, dtype=np.float32)
        return table[np.arange(codes.shape[1])[None, :], codes].sum(axis=1)

    @property
    def bytes_per_vector(self) -> int:
        return self.p.subspaces

    @property
    def compression(self) -> float:
        return (self.dim * 4) / self.bytes_per_vector

    def snapshot(self) -> dict[str, float]:
        return {"subspaces": self.p.subspaces, "bits": self.p.bits, "sub_dim": self.sub_dim,
                "trained_on": self.trained_on, "bytes_per_vector": self.bytes_per_vector,
                "compression": round(self.compression, 1), "trained": self.codebooks is not None,
                "opq_iterations": self.p.opq_iterations,
                "residual_error": round(self.residual_error, 4)}


@dataclass
class IvfPqStats:
    probes: int = 0
    lists_scanned: int = 0
    candidates: int = 0
    rescored: int = 0

    def as_dict(self) -> dict[str, float]:
        return {"probes": self.probes, "lists_scanned": self.lists_scanned,
                "candidates": self.candidates, "rescored": self.rescored,
                "avg_lists": round(self.lists_scanned / self.probes, 2) if self.probes else 0.0}


class IvfPqIndex:
    """Coarse IVF partitioning + PQ residual codes + exact rescoring.

    The coarse quantizer narrows the search to `nprobe` cells, PQ ranks inside
    them without decompressing, and only the surviving handful are rescored
    against full precision. Three tiers of accuracy, each an order of magnitude
    cheaper than the one above it.
    """

    def __init__(self, dim: int, lists: int = 16, nprobe: int = 3,
                 params: PqParams | None = None, target_recall: float = 0.95) -> None:
        self.dim = dim
        self.lists = lists
        self.nprobe = nprobe
        self.target_recall = target_recall
        self.calibrated_depth = 0
        self.calibration: dict[str, float] = {}
        self.pq = ProductQuantizer(dim, params)
        self.coarse: np.ndarray | None = None
        self.postings: dict[int, list[str]] = {}
        self.codes: dict[str, np.ndarray] = {}
        self.cell_of: dict[str, int] = {}
        self.full: dict[str, np.ndarray] = {}         # kept for rescoring
        self.stats = IvfPqStats()
        self.trained = False

    def train(self, vectors: np.ndarray) -> "IvfPqIndex":
        vectors = np.asarray(vectors, dtype=np.float32)
        self.coarse = kmeans(vectors, min(self.lists, max(len(vectors), 1)), seed=5)
        residuals = vectors - self.coarse[self._assign(vectors)]
        self.pq.train(residuals)                       # PQ on residuals, not raw vectors
        self.postings = {i: [] for i in range(len(self.coarse))}
        self.trained = True
        self.calibrate(vectors)
        return self

    def calibrate(self, vectors: np.ndarray, k: int = 10, samples: int = 24) -> dict[str, float]:
        """Measure the rescore depth this corpus actually needs.

        A fixed `rescore = 4k` is a guess, and on a tightly clustered corpus it
        is a bad one: quantization error exceeds the gaps between true
        neighbours, so the right answers sit at rank 200+ in the approximate
        ordering even though the ADC ranking correlates at 0.995.

        So we measure instead of guessing. Sample queries from the training
        set, find the depth at which the true top-k have been recovered to the
        target rate, and use that. Scanning cheap codes deeply and rescoring a
        few hundred exactly is still an order of magnitude less work than
        reading every full-precision vector — that is the entire point of the
        cold tier.
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if len(vectors) < k * 2:
            self.calibrated_depth = k * 4
            return {}
        rng = np.random.default_rng(11)
        probes = vectors[rng.choice(len(vectors), size=min(samples, len(vectors)), replace=False)]
        cells_of = self._assign(vectors)
        codes = self.pq.encode(vectors - self.coarse[cells_of])
        bases = self.coarse @ probes.T

        # 1. nprobe: a true neighbour in an unprobed cell is unrecoverable at
        #    any rescore depth, so raise nprobe until the cells we visit
        #    actually contain the answers. Uniformly distributed corpora have
        #    no cluster structure for IVF to exploit and need far more probes
        #    than clustered ones — which is exactly why this is measured.
        required: list[int] = []
        for query in probes:
            true_top = np.argsort(-(vectors @ query))[:k]
            ranked_cells = list(np.argsort(-(self.coarse @ query)))
            rank_of = {int(c): i for i, c in enumerate(ranked_cells)}
            required.append(max(rank_of.get(int(cells_of[t]), len(ranked_cells)) for t in true_top) + 1)
        required.sort()
        self.nprobe = max(self.nprobe, int(required[min(len(required) - 1,
                                                        int(self.target_recall * len(required)))]))
        self.nprobe = min(self.nprobe, len(self.coarse))

        needed: list[int] = []
        for column, query in enumerate(probes):
            true_top = set(np.argsort(-(vectors @ query))[:k].tolist())
            table = self.pq.lookup_table(query)
            approx = self.pq.score(table, codes) + bases[self._assign(vectors), column]
            order = np.argsort(-approx)
            found, depth = 0, len(order)
            for position, candidate in enumerate(order, start=1):
                if int(candidate) in true_top:
                    found += 1
                    if found >= max(1, int(round(k * self.target_recall))):
                        depth = position
                        break
            needed.append(depth)

        needed.sort()
        index = min(len(needed) - 1, int(self.target_recall * len(needed)))
        self.calibrated_depth = int(needed[index])
        self.calibration = {
            "target_recall": self.target_recall, "k": k,
            "nprobe": self.nprobe, "lists": len(self.coarse),
            "probe_fraction": round(self.nprobe / max(len(self.coarse), 1), 3),
            "depth_p50": float(needed[len(needed) // 2]),
            "depth_chosen": float(self.calibrated_depth),
            "corpus": len(vectors),
            "scan_fraction": round(self.calibrated_depth / len(vectors), 4),
        }
        return self.calibration

    def _assign(self, vectors: np.ndarray) -> np.ndarray:
        return _assign_chunked(vectors, self.coarse)

    def add(self, point_id: str, vector: np.ndarray) -> None:
        if not self.trained:
            raise RuntimeError("index is not trained")
        vector = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        cell = int(self._assign(vector)[0])
        if point_id in self.cell_of:
            self.remove(point_id)
        residual = vector - self.coarse[cell]
        self.codes[point_id] = self.pq.encode(residual)[0]
        self.cell_of[point_id] = cell
        self.postings.setdefault(cell, []).append(point_id)
        self.full[point_id] = vector.reshape(-1).copy()

    def remove(self, point_id: str) -> bool:
        cell = self.cell_of.pop(point_id, None)
        if cell is None:
            return False
        self.postings[cell] = [p for p in self.postings[cell] if p != point_id]
        self.codes.pop(point_id, None)
        self.full.pop(point_id, None)
        return True

    def search(self, query: np.ndarray, k: int, nprobe: int | None = None,
               rescore: int | None = None) -> list[tuple[str, float]]:
        if not self.trained or not self.codes:
            return []
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        self.stats.probes += 1
        nprobe = nprobe or self.nprobe

        cell_scores = self.coarse @ query
        cells = np.argsort(-cell_scores)[:nprobe]
        self.stats.lists_scanned += len(cells)

        ids: list[str] = []
        for cell in cells:
            ids.extend(self.postings.get(int(cell), []))
        if not ids:
            return []
        self.stats.candidates += len(ids)

        codes = np.vstack([self.codes[i] for i in ids])
        # residual ADC + the coarse centroid's own contribution
        approx = np.zeros(len(ids), dtype=np.float32)
        table = self.pq.lookup_table(query)          # one table serves every cell
        for cell in cells:
            members = [i for i, pid in enumerate(ids) if self.cell_of[pid] == int(cell)]
            if not members:
                continue
            base = float(self.coarse[int(cell)] @ query)
            approx[members] = base + self.pq.score(table, codes[members])

        depth = max(k, self.calibrated_depth, (rescore or 0) * k)
        shortlist = np.argsort(-approx)[:depth]
        exact = np.array([float(self.full[ids[i]] @ query) for i in shortlist], dtype=np.float32)
        self.stats.rescored += len(shortlist)
        order = np.argsort(-exact)[:k]
        return [(ids[shortlist[i]], float(exact[i])) for i in order]

    def snapshot(self) -> dict[str, object]:
        return {"lists": len(self.postings), "nprobe": self.nprobe, "points": len(self.codes),
                "calibrated_depth": self.calibrated_depth, "calibration": self.calibration,
                "pq": self.pq.snapshot(), **self.stats.as_dict()}
