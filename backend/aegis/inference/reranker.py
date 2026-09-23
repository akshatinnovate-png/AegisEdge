"""Cross-encoder reranker.

Bi-encoder recall is cheap and imprecise; the cross-encoder sees query and
document together and fixes the ordering. Runs locally on int8 — escalating
every query to the cloud for reranking would defeat the entire premise.
"""
from __future__ import annotations

from pathlib import Path

from ..core.metrics import METRICS
from .onnx_runtime import OnnxSession, tokenize


class Reranker:
    role = "reranker"

    def __init__(self, name: str, model_dir: str = "models", variant: str = "int8-dynamic") -> None:
        self.name = name
        path = Path(model_dir) / f"{name}.{variant}.onnx"
        self.session = OnnxSession(self.role, path if path.exists() else None, 64, Path(model_dir) / ".cache")
        self.calls = 0

    def _pair_score(self, query: str, text: str) -> float:
        """Lexical-overlap cross-attention stand-in when no graph is present.

        Coverage of the query by the document, weighted by term rarity within
        the pair, plus a mild proximity bonus for adjacent query terms.
        """
        q = [t for t in tokenize(query)]
        d = tokenize(text)
        if not q or not d:
            return 0.0
        dset = set(d)
        covered = sum(1 for t in q if t in dset) / len(q)
        bigrams = {f"{a}_{b}" for a, b in zip(d, d[1:])}
        proximity = sum(1 for a, b in zip(q, q[1:]) if f"{a}_{b}" in bigrams) / max(len(q) - 1, 1)
        brevity = min(1.0, 24 / max(len(d), 1)) * 0.15
        return round(0.68 * covered + 0.22 * proximity + brevity, 6)

    def rerank(self, query: str, candidates: list[tuple[str, str, float]], top_k: int) -> list[tuple[str, float]]:
        """candidates: (id, text, retrieval_score) -> reranked (id, score)."""
        self.calls += 1
        with METRICS.timer("inference.rerank_ms"):
            if self.session.session is not None:  # pragma: no cover - needs a real graph
                try:
                    import numpy as np

                    vectors = self.session.encode([f"{query} [SEP] {text}" for _, text, _ in candidates])
                    qv = self.session.encode([query])[0]
                    scores = (vectors @ qv).tolist()
                except Exception:
                    scores = [self._pair_score(query, text) for _, text, _ in candidates]
            else:
                scores = [self._pair_score(query, text) for _, text, _ in candidates]
        # retrieval score is evidence too; the reranker refines, it does not replace
        fused = [
            (pid, round(0.75 * s + 0.25 * retrieval, 6))
            for (pid, _, retrieval), s in zip(candidates, scores)
        ]
        fused.sort(key=lambda x: -x[1])
        return fused[:top_k]

    def snapshot(self) -> dict[str, object]:
        hist = METRICS.histograms.get("inference.rerank_ms")
        return {"model": self.name, "calls": self.calls, "session": self.session.snapshot(),
                "latency_ms": hist.snapshot() if hist else {}}
