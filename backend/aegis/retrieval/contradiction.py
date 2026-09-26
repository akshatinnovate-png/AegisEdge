"""Contradiction detection.

Two memories can be similar in embedding space and mean opposite things —
"pressure below 1.8 bar is a hard stop" versus "the 1.8 bar limit was lifted".
High similarity plus an opposing polarity marker is the cheap on-device signal
for that, and the newer memory supersedes the older rather than deleting it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from ..inference.onnx_runtime import tokenize
from ..inference.sparse import _STOP
from ..memory.schema import MemoryPoint

NEGATIONS = re.compile(
    r"(?i)\b(not|no longer|never|without|lifted|revoked|cancelled|canceled|"
    r"reversed|disabled|removed|superseded|obsolete|deprecated|rescinded)\b"
)
ANTONYMS = [
    ("above", "below"), ("increase", "decrease"), ("open", "closed"),
    ("enable", "disable"), ("start", "stop"), ("raise", "lower"),
    ("allow", "deny"), ("on", "off"), ("up", "down"),
]


@dataclass(slots=True)
class Contradiction:
    older_id: str
    newer_id: str
    similarity: float
    signals: list[str]

    def as_dict(self) -> dict[str, object]:
        return {"older": self.older_id, "newer": self.newer_id,
                "similarity": round(self.similarity, 4), "signals": self.signals}


class ContradictionDetector:
    def __init__(self, threshold: float = 0.45) -> None:
        self.threshold = threshold
        self.found = 0

    @staticmethod
    def _polarity_signals(a: str, b: str) -> list[str]:
        signals: list[str] = []
        if bool(NEGATIONS.search(a)) != bool(NEGATIONS.search(b)):
            signals.append("negation_flip")
        lower_a, lower_b = a.lower(), b.lower()
        for left, right in ANTONYMS:
            in_a = re.search(rf"\b{left}\b", lower_a) and re.search(rf"\b{right}\b", lower_b)
            in_b = re.search(rf"\b{right}\b", lower_a) and re.search(rf"\b{left}\b", lower_b)
            if in_a or in_b:
                signals.append(f"antonym:{left}/{right}")
        return signals

    @staticmethod
    def _lexical_overlap(a: str, b: str) -> float:
        """Overlap coefficient over content tokens.

        Jaccard would punish the common case here — a short correction
        ("the 1.8 bar stop was lifted") against a longer original — because
        the union grows with the difference in length. The overlap
        coefficient measures how much of the *smaller* statement the other
        one covers, which is the question being asked.
        """
        ta = {t for t in tokenize(a) if t not in _STOP}
        tb = {t for t in tokenize(b) if t not in _STOP}
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / min(len(ta), len(tb))

    def relatedness(self, candidate: MemoryPoint, other: MemoryPoint) -> float:
        """Are these two memories even about the same thing?

        Cosine alone is encoder-dependent: a quantized on-device embedder
        compresses the similarity range, and a contradiction that is obvious
        lexically would be missed. Taking the stronger of the two signals
        keeps detection stable across every variant the governor may swap in.
        """
        cosine = 0.0
        if candidate.has_dense and other.has_dense:
            a = np.asarray(candidate.dense, dtype=np.float32)
            b = np.asarray(other.dense, dtype=np.float32)
            denominator = float(np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
            cosine = float(a @ b / denominator)
        return max(cosine, self._lexical_overlap(candidate.text, other.text))

    def check(self, candidate: MemoryPoint, others: list[MemoryPoint]) -> list[Contradiction]:
        out: list[Contradiction] = []
        for other in others:
            if other.id == candidate.id or other.superseded_by:
                continue
            similarity = self.relatedness(candidate, other)
            if similarity < self.threshold:
                continue
            signals = self._polarity_signals(candidate.text, other.text)
            if not signals:
                continue
            older, newer = (other, candidate) if other.created_at <= candidate.created_at else (candidate, other)
            out.append(Contradiction(older.id, newer.id, similarity, signals))
            self.found += 1
        return out
