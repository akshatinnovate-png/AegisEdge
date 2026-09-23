"""Query understanding.

An operator with gloves on types "colent presure bay3". A cloud stack throws
an LLM at that; an edge node has milliseconds and no network, so it needs
cheap, local, corpus-derived machinery:

* a **BK-tree** over the corpus vocabulary for metric spelling repair —
  correcting only to words the node has actually seen, which beats a generic
  dictionary because the vocabulary is domain-specific ("gantry", part codes);
* **co-occurrence expansion** learned from ingested text, so "coolant" pulls
  in "pressure" and "bar" without an embedding round trip;
* **acronym and unit normalisation**, because 4.2mm/s and "4.2 mm/s" are the
  same reading;
* **intent + filter extraction**, turning "sensor readings from last hour"
  into a payload filter the planner can actually use.
"""
from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..inference.onnx_runtime import tokenize
from ..inference.sparse import _STOP

UNIT = re.compile(r"(?i)\b(\d+(?:\.\d+)?)\s*(mm/s|mm|bar|nm|rpm|°?c|hz|kw|a|v|ms|s|min|h)\b")
RELATIVE_TIME = re.compile(
    r"(?i)\b(?:in the |over the |from the |during the )?(?:last|past|previous)\s+"
    r"(\d+)?\s*(second|minute|hour|day|week|month)s?\b")
# Frequent English words the corpus may never contain. Correcting these to a
# similar-looking domain term ("last" -> "mast") corrupts the query while
# looking confident, which is worse than leaving a typo alone.
PROTECTED = frozenset(
    "last next first this that these those from over under about after before during "
    "there where when what which while since until between against through many much "
    "more most some other another every each both few less least same than then only "
    "just also very such even still back down left right near far high low long short "
    "time hour day week month year today yesterday tomorrow night morning shift line "
    "show find list give tell make take come know think want need said says".split()
)

COLLECTION_HINT = {
    "sensor": ("sensor", "reading", "telemetry", "measurement", "vibration", "temperature"),
    "procedural": ("procedure", "steps", "how", "recovery", "fix", "repair", "checklist"),
    "episodic": ("happened", "yesterday", "shift", "event", "when", "log"),
    "semantic": ("rule", "policy", "threshold", "limit", "definition", "means"),
}


def levenshtein(a: str, b: str, cap: int = 3) -> int:
    """Bounded edit distance — abandons early once the cap is exceeded."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        best = i
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            value = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            current.append(value)
            best = min(best, value)
        if best > cap:
            return cap + 1
        previous = current
    return previous[-1]


class BKTree:
    """Metric tree over edit distance: finds near words without scanning the lexicon."""

    def __init__(self) -> None:
        self.root: str | None = None
        self.children: dict[str, dict[int, str]] = {}
        self.size = 0
        self.probes = 0
        self.nodes_visited = 0

    def add(self, word: str) -> None:
        if self.root is None:
            self.root, self.children[word] = word, {}
            self.size = 1
            return
        node = self.root
        while True:
            distance = levenshtein(word, node, cap=32)
            if distance == 0:
                return
            edges = self.children.setdefault(node, {})
            if distance in edges:
                node = edges[distance]
                continue
            edges[distance] = word
            self.children.setdefault(word, {})
            self.size += 1
            return

    def search(self, word: str, tolerance: int = 2) -> list[tuple[str, int]]:
        if self.root is None:
            return []
        self.probes += 1
        found: list[tuple[str, int]] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            self.nodes_visited += 1
            # The branch test below is the triangle inequality, which needs the
            # TRUE distance. Using the cheap bounded distance here would clamp
            # far-away nodes to cap+1 and prune the very subtrees that hold the
            # matches — silently turning the tree into a no-op.
            distance = levenshtein(word, node, cap=64)
            if distance <= tolerance:
                found.append((node, distance))
            for edge, child in self.children.get(node, {}).items():
                # triangle inequality: only these branches can contain a match
                if distance - tolerance <= edge <= distance + tolerance:
                    stack.append(child)
        found.sort(key=lambda x: (x[1], -len(x[0])))
        return found


@dataclass
class Understanding:
    original: str
    normalized: str
    corrections: dict[str, str] = field(default_factory=dict)
    expansions: list[str] = field(default_factory=list)
    intent: str = "semantic"
    collection: str = "*"
    filters: dict[str, Any] = field(default_factory=dict)
    units: list[dict[str, Any]] = field(default_factory=list)
    ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"original": self.original, "normalized": self.normalized,
                "corrections": self.corrections, "expansions": self.expansions,
                "intent": self.intent, "collection": self.collection,
                "filters": self.filters, "units": self.units, "ms": round(self.ms, 3)}


class QueryUnderstanding:
    def __init__(self, min_term_frequency: int = 1) -> None:
        self.vocabulary: Counter[str] = Counter()
        self.tree = BKTree()
        self.cooccurrence: dict[str, Counter[str]] = defaultdict(Counter)
        self.documents = 0
        # A term enters the correction lexicon once the corpus has shown it
        # this many times. 1 suits an edge node, whose corpus starts small and
        # is the only authority on its own domain vocabulary; raise it where
        # ingest is noisy enough to contain misspellings of its own.
        self.min_term_frequency = min_term_frequency
        self.queries = 0

    # -- learning from the corpus ----------------------------------------

    def observe(self, text: str) -> None:
        """Every ingested memory teaches the vocabulary and the co-occurrences."""
        terms = [t for t in tokenize(text) if t not in _STOP and len(t) > 2]
        if not terms:
            return
        self.documents += 1
        for term in terms:
            self.vocabulary[term] += 1
            if self.vocabulary[term] == self.min_term_frequency:
                self.tree.add(term)                 # only words seen more than once
        unique = set(terms)
        for term in unique:
            for other in unique:
                if other != term:
                    self.cooccurrence[term][other] += 1

    # -- query time -------------------------------------------------------

    def _correct(self, token: str) -> str | None:
        if token in self.vocabulary or len(token) < 4 or token.isdigit():
            return None
        if token in PROTECTED or token in _STOP:
            return None                       # a real word, just not in this corpus
        tolerance = 1 if len(token) <= 5 else 2
        matches = self.tree.search(token, tolerance)
        if not matches:
            return None
        best = max(matches, key=lambda m: (-m[1], self.vocabulary[m[0]]))
        return best[0] if self.vocabulary[best[0]] >= self.min_term_frequency else None

    def _expand(self, terms: list[str], limit: int = 3) -> list[str]:
        """Pointwise-mutual-information style expansion from the local corpus."""
        scores: Counter[str] = Counter()
        present = set(terms)
        for term in terms:
            partners = self.cooccurrence.get(term)
            if not partners:
                continue
            total = sum(partners.values()) or 1
            for other, count in partners.most_common(24):
                if other in present or other in _STOP or other in PROTECTED:
                    continue
                if not any(c.isalpha() for c in other) or len(other) < 3:
                    continue                  # bare numbers expand nothing useful
                probability = count / total
                rarity = math.log(1 + self.documents / max(self.vocabulary[other], 1))
                scores[other] += probability * rarity
        return [term for term, _ in scores.most_common(limit)]

    @staticmethod
    def _units(text: str) -> tuple[str, list[dict[str, Any]]]:
        found: list[dict[str, Any]] = []

        def normalise(match: re.Match[str]) -> str:
            value, unit = match.group(1), match.group(2).lower().replace("°", "")
            found.append({"value": float(value), "unit": unit})
            return f"{value} {unit}"

        return UNIT.sub(normalise, text), found

    @staticmethod
    def _temporal(text: str) -> dict[str, Any]:
        match = RELATIVE_TIME.search(text)
        if not match:
            return {}
        amount = int(match.group(1) or 1)
        seconds = {"second": 1, "minute": 60, "hour": 3600,
                   "day": 86400, "week": 604800, "month": 2592000}[match.group(2).lower()]
        return {"ts": {"gte": time.time() - amount * seconds}}

    def _intent(self, terms: list[str]) -> tuple[str, str]:
        best, score = "semantic", 0
        for collection, markers in COLLECTION_HINT.items():
            hits = sum(1 for t in terms if t in markers)
            if hits > score:
                best, score = collection, hits
        return (best, best) if score else ("semantic", "*")

    def analyse(self, query: str) -> Understanding:
        t0 = time.perf_counter()
        self.queries += 1
        normalized, units = self._units(query)
        tokens = tokenize(normalized)

        corrections: dict[str, str] = {}
        repaired: list[str] = []
        for token in tokens:
            fixed = self._correct(token)
            if fixed:
                corrections[token] = fixed
                repaired.append(fixed)
            else:
                repaired.append(token)

        content = [t for t in repaired if t not in _STOP and len(t) > 2]
        expansions = self._expand(content)
        intent, collection = self._intent(repaired)
        filters = self._temporal(query)
        if collection != "*":
            filters = {**filters, "collection": collection}

        rewritten = " ".join(repaired)
        if expansions:
            rewritten = f"{rewritten} {' '.join(expansions)}"

        return Understanding(
            original=query, normalized=rewritten, corrections=corrections,
            expansions=expansions, intent=intent, collection=collection,
            filters=filters, units=units, ms=(time.perf_counter() - t0) * 1000,
        )

    def snapshot(self) -> dict[str, Any]:
        return {"vocabulary": len(self.vocabulary), "documents": self.documents,
                "bk_tree_nodes": self.tree.size, "bk_probes": self.tree.probes,
                "avg_nodes_visited": round(self.tree.nodes_visited / self.tree.probes, 1)
                if self.tree.probes else 0.0,
                "cooccurrence_terms": len(self.cooccurrence), "queries": self.queries}
