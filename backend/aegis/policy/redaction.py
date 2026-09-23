"""Reversible redaction with a local-only vault.

The cloud sees `<EMAIL:7f31>`; the device can still resolve that token to the
original because the mapping never leaves local storage. Retrieval quality is
preserved on-device without exporting the entity.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from .engine import PolicyEngine  # noqa: F401  (typing convenience)

MASKS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+\d{1,3}[\s-]?)?(?:\d[\s-]?){9,13}\d(?!\d)")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("ID", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("GEO", re.compile(r"-?\d{1,3}\.\d{4,},\s*-?\d{1,3}\.\d{4,}")),
    ("SECRET", re.compile(r"(?i)(?:api[_ -]?key|token|password|secret)\s*[=:]\s*\S+")),
    ("OPERATOR", re.compile(r"(?i)\boperator\s+(?:id\s+)?[a-z]?\d{2,}\b")),
]


@dataclass
class RedactionVault:
    """Token -> original. Never serialized into any egress payload."""
    mapping: dict[str, str] = field(default_factory=dict)
    reverse: dict[str, str] = field(default_factory=dict)
    redactions: int = 0

    def token_for(self, kind: str, value: str) -> str:
        if value in self.reverse:
            return self.reverse[value]
        digest = hashlib.blake2b(value.encode("utf-8"), digest_size=2).hexdigest()
        token = f"<{kind}:{digest}>"
        self.mapping[token] = value
        self.reverse[value] = token
        return token

    def redact(self, text: str) -> tuple[str, list[str]]:
        applied: list[str] = []
        out = text
        for kind, pattern in MASKS:
            def _sub(match: re.Match[str]) -> str:
                applied.append(kind)
                self.redactions += 1
                return self.token_for(kind, match.group(0))
            out = pattern.sub(_sub, out)
        return out, applied

    def resolve(self, text: str) -> str:
        """Local-only: put the originals back for on-device display."""
        for token, value in self.mapping.items():
            text = text.replace(token, value)
        return text

    def snapshot(self) -> dict[str, int]:
        return {"entries": len(self.mapping), "redactions": self.redactions}
