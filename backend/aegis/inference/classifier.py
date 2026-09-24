"""On-device sensitivity / PII classifier.

Runs before anything is written, never after. A point that is classified
`restricted` is never eligible to leave the device, so classification has to
happen on the ingest path where it cannot be bypassed.

Being on the ingest path is exactly what makes its worst case dangerous. A
stress run wedged this node for over sixteen minutes at 100% CPU on a single
`remember()` call, and the culprit was one unanchored regex: `[\w.+-]+@...`
matches a megabyte of word characters greedily, fails to find the `@`,
backtracks the whole way, and the engine then retries from the next offset,
and the next. Quadratic — measured at exactly 4x the time for 2x the input,
which extrapolates to 79 minutes for a 1 MB write. One unauthenticated ingest
takes the device off the air.

Two defences, because the pattern fix alone protects against the bug that was
found rather than the class it belongs to:

* every pattern is anchored so it cannot restart inside a run it has already
  rejected; and
* no pattern is ever handed an unbounded string — the scan runs in windows
  with an overlap wider than any credential, so cost stays linear in the input
  even if a future edit reintroduces a greedy prefix.

Truncation was considered and rejected. This is a security control: a secret
at offset two megabytes must not escape classification because scanning it
would have been inconvenient.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..memory.schema import Sensitivity

# Window and overlap for the chunked scan. The overlap must comfortably exceed
# the longest match any pattern can produce, or a credential straddling a
# window boundary would be missed — which is the one failure mode a PII
# classifier may not have.
SCAN_WINDOW = 64 * 1024
SCAN_OVERLAP = 1024

PATTERNS: list[tuple[str, re.Pattern[str], Sensitivity]] = [
    # The lookbehind is load-bearing, not cosmetic: without it this pattern is
    # quadratic on any long run of word characters. See the module docstring.
    ("email", re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+\.[\w.]+"), Sensitivity.RESTRICTED),
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

    def __init__(self, window: int = SCAN_WINDOW, overlap: int = SCAN_OVERLAP) -> None:
        self.classified = 0
        self.by_class: dict[str, int] = {s.value: 0 for s in Sensitivity}
        self.window = max(int(window), 1024)
        self.overlap = min(max(int(overlap), 64), self.window // 2)
        self.chunked = 0

    def _windows(self, text: str):
        """Yield overlapping slices, so no pattern sees an unbounded string."""
        if len(text) <= self.window:
            yield text
            return
        self.chunked += 1
        step = self.window - self.overlap
        for start in range(0, len(text), step):
            yield text[start:start + self.window]
            if start + self.window >= len(text):
                break

    def classify(self, text: str) -> Classification:
        self.classified += 1
        hits: list[str] = []
        level = Sensitivity.PUBLIC
        windows = list(self._windows(text))
        for name, pattern, sensitivity in PATTERNS:
            if any(pattern.search(window) for window in windows):
                hits.append(name)
                if sensitivity.rank > level.rank:
                    level = sensitivity
        if not hits and len(text.split()) > 3:
            level = Sensitivity.INTERNAL          # default posture: not public
        confidence = 0.62 + 0.12 * len(hits) if hits else 0.55
        self.by_class[level.value] += 1
        return Classification(level, hits, min(confidence, 0.99))

    def snapshot(self) -> dict[str, object]:
        return {"classified": self.classified, "by_class": dict(self.by_class),
                "scan_window": self.window, "scan_overlap": self.overlap,
                "chunked_scans": self.chunked}
