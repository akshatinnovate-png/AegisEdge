"""On-device sensitivity / PII classifier.

Runs before anything is written, never after. A point that is classified
`restricted` is never eligible to leave the device, so classification has to
happen on the ingest path where it cannot be bypassed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..memory.schema import Sensitivity

PATTERNS: list[tuple[str, re.Pattern[str], Sensitivity]] = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), Sensitivity.RESTRICTED),
    ("phone", re.compile(r"(?<!\d)(?:\+\d{1,3}[\s-]?)?(?:\d[\s-]?){9,13}\d(?!\d)"), Sensitivity.RESTRICTED),
    ("national_id", re.compile(r"\b\d{3}-\d{2}-\d{4}\b|\b\d{4}\s?\d{4}\s?\d{4}\b"), Sensitivity.RESTRICTED),
    ("card", re.compile(r"\b(?:\d[ -]?){13,16}\b"), Sensitivity.RESTRICTED),
    ("coords", re.compile(r"-?\d{1,3}\.\d{4,},\s*-?\d{1,3}\.\d{4,}"), Sensitivity.SENSITIVE),
    ("credential", re.compile(r"(?i)\b(password|api[_ -]?key|secret|token|bearer)\b"), Sensitivity.RESTRICTED),
    ("operator", re.compile(r"(?i)\boperator\s+(id\s+)?[a-z]?\d{2,}\b"), Sensitivity.SENSITIVE),
    ("internal", re.compile(r"(?i)\b(internal|confidential|do not share)\b"), Sensitivity.SENSITIVE),
]


@dataclass(slots=True)
class Classification:
    sensitivity: Sensitivity
    signals: list[str]
    confidence: float

    def as_dict(self) -> dict[str, object]:
        return {"sensitivity": self.sensitivity.value, "signals": self.signals,
                "confidence": round(self.confidence, 3)}


class SensitivityClassifier:
    """Rule ensemble standing in for the ONNX head; same interface either way."""

    def __init__(self) -> None:
        self.classified = 0
        self.by_class: dict[str, int] = {s.value: 0 for s in Sensitivity}

    def classify(self, text: str) -> Classification:
        self.classified += 1
        hits: list[str] = []
        level = Sensitivity.PUBLIC
        for name, pattern, sensitivity in PATTERNS:
            if pattern.search(text):
                hits.append(name)
                if sensitivity.rank > level.rank:
                    level = sensitivity
        if not hits and len(text.split()) > 3:
            level = Sensitivity.INTERNAL          # default posture: not public
        confidence = 0.62 + 0.12 * len(hits) if hits else 0.55
        self.by_class[level.value] += 1
        return Classification(level, hits, min(confidence, 0.99))

    def snapshot(self) -> dict[str, object]:
        return {"classified": self.classified, "by_class": dict(self.by_class)}
