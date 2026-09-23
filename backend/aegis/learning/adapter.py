"""On-device retrieval adapter.

A shipped embedder knows English; it does not know that on *this* line
"chatter" means bearing noise rather than conversation. The fix is not to
fine-tune a transformer on a kiosk — it is a small low-rank adapter over the
embedding space, trained from the feedback the node already collects.

    score(q, d) = qᵀd + α · (Uq)ᵀ(Vd)

U and V are rank-r projections (r ≈ 16), trained by contrastive updates: pull
a clicked result toward its query, push an explicitly rejected one away. Two
matrices of 16x384 floats — about 50 KB — and they train in microseconds per
example, which is what makes this feasible on the device that collected the
feedback rather than in a cloud job that never sees it.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class TrainingExample:
    query: np.ndarray
    positive: np.ndarray
    negative: np.ndarray | None = None
    weight: float = 1.0
    at: float = field(default_factory=time.time)


@dataclass
class AdapterStats:
    updates: int = 0
    examples: int = 0
    loss_sum: float = 0.0
    last_loss: float = 0.0
    applied: int = 0
    reorders: int = 0

    def as_dict(self) -> dict[str, float]:
        return {"updates": self.updates, "examples": self.examples,
                "avg_loss": round(self.loss_sum / self.updates, 5) if self.updates else 0.0,
                "last_loss": round(self.last_loss, 5), "applied": self.applied,
                "reorders": self.reorders}


class RetrievalAdapter:
    def __init__(self, dim: int, rank: int = 16, alpha: float = 0.35,
                 learning_rate: float = 0.05, seed: int = 23) -> None:
        self.dim = dim
        self.rank = rank
        self.alpha = alpha
        self.lr = learning_rate
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(dim)
        self.u = (rng.normal(size=(rank, dim)) * scale).astype(np.float32)
        self.v = (rng.normal(size=(rank, dim)) * scale).astype(np.float32)
        self.stats = AdapterStats()
        self.version = 0
        self.frozen = False

    # -- inference --------------------------------------------------------

    def delta(self, query: np.ndarray, documents: np.ndarray) -> np.ndarray:
        """The adapter's contribution to the score — added, never replacing."""
        query_projection = self.u @ np.asarray(query, dtype=np.float32)
        document_projection = np.asarray(documents, dtype=np.float32) @ self.v.T
        return self.alpha * (document_projection @ query_projection)

    def rescore(self, query: np.ndarray, candidates: list[tuple[str, float, np.ndarray]]
                ) -> list[tuple[str, float, float]]:
        """Returns (id, adapted_score, delta) preserving the base ordering info."""
        if not candidates:
            return []
        self.stats.applied += 1
        matrix = np.vstack([c[2] for c in candidates]).astype(np.float32)
        adjustments = self.delta(query, matrix)
        before = [c[0] for c in sorted(candidates, key=lambda c: -c[1])]
        out = [(pid, float(base + adj), float(adj))
               for (pid, base, _), adj in zip(candidates, adjustments)]
        out.sort(key=lambda row: -row[1])
        if [row[0] for row in out] != before:
            self.stats.reorders += 1
        return out

    # -- training ---------------------------------------------------------

    def _grad(self, example: TrainingExample) -> tuple[np.ndarray, np.ndarray, float]:
        """Contrastive hinge on the adapter term only; the base score is fixed."""
        query = np.asarray(example.query, dtype=np.float32)
        positive = np.asarray(example.positive, dtype=np.float32)
        uq = self.u @ query
        vp = self.v @ positive
        positive_score = float(uq @ vp)

        if example.negative is None:
            loss = max(0.0, 1.0 - positive_score)
            if loss == 0.0:
                return np.zeros_like(self.u), np.zeros_like(self.v), 0.0
            grad_u = -np.outer(vp, query)
            grad_v = -np.outer(uq, positive)
            return grad_u, grad_v, loss

        negative = np.asarray(example.negative, dtype=np.float32)
        vn = self.v @ negative
        negative_score = float(uq @ vn)
        margin = 0.2
        loss = max(0.0, margin - (positive_score - negative_score))
        if loss == 0.0:
            return np.zeros_like(self.u), np.zeros_like(self.v), 0.0
        grad_u = -np.outer(vp - vn, query)
        grad_v = -np.outer(uq, positive) + np.outer(uq, negative)
        return grad_u, grad_v, loss

    def learn(self, examples: list[TrainingExample], clip: float = 1.0) -> float:
        """One SGD step over a small batch, with gradient clipping."""
        if self.frozen or not examples:
            return 0.0
        grad_u = np.zeros_like(self.u)
        grad_v = np.zeros_like(self.v)
        total = 0.0
        for example in examples:
            du, dv, loss = self._grad(example)
            grad_u += du * example.weight
            grad_v += dv * example.weight
            total += loss * example.weight
        scale = max(1.0, float(np.linalg.norm(grad_u) + np.linalg.norm(grad_v)) / clip)
        self.u -= self.lr * grad_u / (scale * len(examples))
        self.v -= self.lr * grad_v / (scale * len(examples))
        self.stats.updates += 1
        self.stats.examples += len(examples)
        self.stats.last_loss = total / len(examples)
        self.stats.loss_sum += self.stats.last_loss
        self.version += 1
        return self.stats.last_loss

    # -- federation -------------------------------------------------------

    def parameters(self) -> np.ndarray:
        return np.concatenate([self.u.ravel(), self.v.ravel()]).astype(np.float32)

    def load_parameters(self, flat: np.ndarray) -> None:
        split = self.rank * self.dim
        self.u = flat[:split].reshape(self.rank, self.dim).astype(np.float32).copy()
        self.v = flat[split:].reshape(self.rank, self.dim).astype(np.float32).copy()
        self.version += 1

    def diff_from(self, baseline: np.ndarray) -> np.ndarray:
        return self.parameters() - np.asarray(baseline, dtype=np.float32)

    # -- persistence ------------------------------------------------------

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({
            "dim": self.dim, "rank": self.rank, "alpha": self.alpha,
            "version": self.version, "u": self.u.tolist(), "v": self.v.tolist(),
        }), encoding="utf-8")

    def load(self, path: Path) -> bool:
        path = Path(path)
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data["dim"] != self.dim or data["rank"] != self.rank:
                return False
            self.u = np.asarray(data["u"], dtype=np.float32)
            self.v = np.asarray(data["v"], dtype=np.float32)
            self.version = int(data.get("version", 0))
            return True
        except Exception:
            return False

    def snapshot(self) -> dict[str, Any]:
        return {"dim": self.dim, "rank": self.rank, "alpha": self.alpha,
                "version": self.version, "frozen": self.frozen,
                "parameters": int(self.u.size + self.v.size),
                "bytes": int(self.u.nbytes + self.v.nbytes), **self.stats.as_dict()}
