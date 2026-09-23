"""Late interaction (ColBERT-style MaxSim).

A single vector per document is a bottleneck: everything the passage says is
averaged into one point, and a long procedure with one relevant step looks
distant from a query about that step. Late interaction keeps one vector per
token and scores by MaxSim — for each query token, the best-matching document
token, summed.

That is more expensive than a dot product, so it runs only on the shortlist
the cheap stages already produced, and the per-token matrices are stored
int8-quantized to keep the memory honest.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.metrics import METRICS
from ..inference.onnx_runtime import tokenize
from ..memory.quantize import ScalarQuantizer


@dataclass
class MaxSimExplanation:
    point_id: str
    score: float
    matches: list[tuple[str, str, float]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.point_id, "maxsim": round(self.score, 4),
                "alignments": [{"query_term": q, "doc_term": d, "sim": round(s, 3)}
                               for q, d, s in self.matches]}


class LateInteractionIndex:
    """Per-token vectors, int8 on disk-free storage, MaxSim at query time."""

    def __init__(self, embedder, max_tokens: int = 48) -> None:
        self.embedder = embedder
        self.max_tokens = max_tokens
        self.codes: dict[str, np.ndarray] = {}
        self.scales: dict[str, np.ndarray] = {}
        self.terms: dict[str, list[str]] = {}
        self.encoded = 0
        self.scored = 0

    def _token_matrix(self, text: str) -> tuple[np.ndarray, list[str]]:
        terms = tokenize(text)[: self.max_tokens]
        if not terms:
            return np.zeros((0, self.embedder.dim), dtype=np.float32), []
        vectors = self.embedder.embed_sync(terms)
        vectors = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms, terms

    def index(self, point_id: str, text: str) -> int:
        matrix, terms = self._token_matrix(text)
        if not len(matrix):
            return 0
        codes, scales = ScalarQuantizer.encode(matrix)     # int8: 4x smaller per token
        self.codes[point_id] = codes
        self.scales[point_id] = scales
        self.terms[point_id] = terms
        self.encoded += 1
        return len(terms)

    def remove(self, point_id: str) -> None:
        self.codes.pop(point_id, None)
        self.scales.pop(point_id, None)
        self.terms.pop(point_id, None)

    def score(self, query: str, point_ids: list[str], explain: bool = False
              ) -> list[MaxSimExplanation]:
        known = [p for p in point_ids if p in self.codes]
        if not known:
            return []
        t0 = time.perf_counter()
        query_matrix, query_terms = self._token_matrix(query)
        if not len(query_matrix):
            return []

        out: list[MaxSimExplanation] = []
        for point_id in known:
            document = ScalarQuantizer.decode(self.codes[point_id], self.scales[point_id])
            similarity = query_matrix @ document.T            # (q_tokens, d_tokens)
            best = similarity.max(axis=1)
            score = float(best.mean())
            matches: list[tuple[str, str, float]] = []
            if explain:
                columns = similarity.argmax(axis=1)
                order = np.argsort(-best)[:4]
                matches = [(query_terms[i], self.terms[point_id][columns[i]], float(best[i]))
                           for i in order]
            out.append(MaxSimExplanation(point_id, score, matches))
            self.scored += 1
        METRICS.observe("retrieval.maxsim_ms", (time.perf_counter() - t0) * 1000)
        out.sort(key=lambda e: -e.score)
        return out

    @property
    def resident_bytes(self) -> int:
        return sum(c.nbytes + self.scales[p].nbytes for p, c in self.codes.items())

    def snapshot(self) -> dict[str, Any]:
        hist = METRICS.histograms.get("retrieval.maxsim_ms")
        tokens = sum(len(t) for t in self.terms.values())
        return {"documents": len(self.codes), "tokens": tokens,
                "avg_tokens": round(tokens / len(self.codes), 1) if self.codes else 0.0,
                "resident_bytes": self.resident_bytes, "encoded": self.encoded,
                "scored": self.scored, "latency_ms": hist.snapshot() if hist else {}}
