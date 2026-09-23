"""Late-interaction reranker.

Bi-encoder recall compresses a whole passage into one vector, which is cheap
and imprecise: a long procedure with one relevant step looks distant from a
query about that step. The reranker scores query and document *token by
token* — ColBERT-style MaxSim, where each query token takes its best match in
the document and those maxima are averaged.

It runs on the same pretrained embedding space as recall, through its own ONNX
graph, on the shortlist only. That is what makes it affordable on a device
with no GPU: the expensive stage sees ten candidates, not ten thousand.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.metrics import METRICS
from .onnx_runtime import OnnxSession


@dataclass(slots=True)
class Alignment:
    query_term: str
    doc_term: str
    similarity: float

    def as_dict(self) -> dict[str, Any]:
        return {"query_term": self.query_term, "doc_term": self.doc_term,
                "sim": round(self.similarity, 4)}


@dataclass
class RerankExplanation:
    point_id: str
    score: float
    alignments: list[Alignment] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.point_id, "maxsim": round(self.score, 4),
                "alignments": [a.as_dict() for a in self.alignments]}


class Reranker:
    role = "reranker"

    def __init__(self, session: OnnxSession, cache_size: int = 2048) -> None:
        self.session = session
        self.name = "late-interaction-maxsim"
        self.calls = 0
        self.cache_size = cache_size
        self._cache: dict[str, tuple[np.ndarray, list[str]]] = {}
        self.cache_hits = 0

    def _tokens(self, key: str, text: str) -> tuple[np.ndarray, list[str]]:
        cached = self._cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        vectors, terms = self.session.token_embeddings(text)
        if len(self._cache) >= self.cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = (vectors, terms)
        return vectors, terms

    def maxsim(self, query: str, candidates: list[tuple[str, str]], explain: bool = False
               ) -> list[RerankExplanation]:
        """candidates: (id, text) -> MaxSim scored, highest first."""
        if not candidates:
            return []
        self.calls += 1
        with METRICS.timer("inference.rerank_ms"):
            query_vectors, query_terms = self._tokens(f"q::{query}", query)
            if not len(query_vectors):
                return []
            out: list[RerankExplanation] = []
            for point_id, text in candidates:
                doc_vectors, doc_terms = self._tokens(point_id, text)
                if not len(doc_vectors):
                    out.append(RerankExplanation(point_id, 0.0))
                    continue
                similarity = query_vectors @ doc_vectors.T
                best = similarity.max(axis=1)
                score = float(best.mean())
                alignments: list[Alignment] = []
                if explain:
                    columns = similarity.argmax(axis=1)
                    for index in np.argsort(-best)[:4]:
                        alignments.append(Alignment(query_terms[index],
                                                    doc_terms[columns[index]],
                                                    float(best[index])))
                out.append(RerankExplanation(point_id, score, alignments))
        out.sort(key=lambda row: -row.score)
        return out

    def rerank(self, query: str, candidates: list[tuple[str, str, float]], top_k: int,
               explain: bool = False) -> list[tuple[str, float]]:
        """candidates: (id, text, retrieval_score) -> reranked (id, blended score)."""
        scored = self.maxsim(query, [(pid, text) for pid, text, _ in candidates], explain)
        retrieval = {pid: score for pid, _, score in candidates}
        # Retrieval score is evidence too; the reranker refines rather than replaces.
        fused = [(row.point_id, round(0.75 * row.score + 0.25 * retrieval.get(row.point_id, 0.0), 6))
                 for row in scored]
        self.last_explanations = {row.point_id: row for row in scored}
        fused.sort(key=lambda row: -row[1])
        return fused[:top_k]

    def snapshot(self) -> dict[str, Any]:
        hist = METRICS.histograms.get("inference.rerank_ms")
        return {"model": self.name, "calls": self.calls, "token_cache": len(self._cache),
                "cache_hits": self.cache_hits, "session": self.session.snapshot(),
                "latency_ms": hist.snapshot() if hist else {}}
