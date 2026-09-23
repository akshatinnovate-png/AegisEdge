"""Sparse encoder (SPLADE-shaped, BM25-backed).

Dense vectors lose rare tokens — part numbers, error codes, the things an
operator actually searches for. The sparse side keeps them, and it is the half
of hybrid retrieval that has to work with no network and no reranker.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter

from .onnx_runtime import tokenize

_STOP = frozenset("the a an of to and or in on at is was for with by from that this it as be".split())


class SparseEncoder:
    def __init__(self, dim: int = 1 << 18, k1: float = 1.2, b: float = 0.75) -> None:
        self.dim = dim
        self.k1 = k1
        self.b = b
        self.doc_freq: Counter[int] = Counter()
        self.docs = 0
        self.total_len = 0

    def term_id(self, token: str) -> int:
        return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest(), "big") % self.dim

    def _terms(self, text: str) -> list[str]:
        tokens = [t for t in tokenize(text) if t not in _STOP and len(t) > 1]
        # keep expanded bigrams: "coolant pressure" is a term of its own
        return tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]

    def fit(self, text: str) -> None:
        """Update corpus statistics — IDF has to be learned on-device."""
        terms = self._terms(text)
        self.docs += 1
        self.total_len += len(terms)
        for term in {self.term_id(t) for t in terms}:
            self.doc_freq[term] += 1

    def idf(self, term: int) -> float:
        n = self.doc_freq.get(term, 0)
        return math.log(1.0 + (self.docs - n + 0.5) / (n + 0.5))

    def encode(self, text: str, fit: bool = False) -> dict[int, float]:
        if fit:
            self.fit(text)
        terms = self._terms(text)
        if not terms:
            return {}
        counts = Counter(self.term_id(t) for t in terms)
        avg_len = (self.total_len / self.docs) if self.docs else len(terms)
        length_norm = self.k1 * (1 - self.b + self.b * len(terms) / max(avg_len, 1.0))
        weights = {
            term: round(self.idf(term) * (tf * (self.k1 + 1)) / (tf + length_norm), 6)
            for term, tf in counts.items()
        }
        # impact pruning: the tail of a sparse vector costs postings and buys nothing
        top = sorted(weights.items(), key=lambda kv: -kv[1])[:64]
        return {t: w for t, w in top if w > 0}

    def snapshot(self) -> dict[str, float]:
        return {
            "vocabulary": len(self.doc_freq),
            "documents": self.docs,
            "avg_doc_len": round(self.total_len / self.docs, 2) if self.docs else 0.0,
            "dim": self.dim,
        }
