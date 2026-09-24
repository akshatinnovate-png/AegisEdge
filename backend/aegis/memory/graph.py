"""Bitemporal knowledge graph over memories.

Vector search answers "what looks like this query". It cannot answer "which
bearings on line 2 were replaced after the torque fault", because that is a
*structural* question, and it cannot answer "what did we believe on Tuesday",
because embeddings have no notion of belief over time.

Two time axes, kept separately, because conflating them loses the ability to
audit:

* **valid time** — when the fact was true in the world.
* **transaction time** — when this node came to believe it, and when it
  stopped.

That distinction is what makes "what did we know on Tuesday, about the state
of the line on Monday" a query rather than an archaeology project — the
question every incident review actually asks. Retraction is a write to the
transaction axis; nothing is deleted, so a belief the node has abandoned is
still inspectable.

Extraction is rule-based and runs on-device in microseconds. It is not an LLM
and does not pretend to be: it covers the entity shapes industrial text
actually contains — part codes, bays, lines, measurements, operators,
procedures — and records its own confidence.
"""
from __future__ import annotations

import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

INFINITY = float("inf")


class EntityType(str, Enum):
    PART = "part"
    ASSET = "asset"
    LOCATION = "location"
    METRIC = "metric"
    PERSON = "person"
    PROCEDURE = "procedure"
    EVENT = "event"
    THRESHOLD = "threshold"
    UNKNOWN = "unknown"


ENTITY_PATTERNS: list[tuple[EntityType, re.Pattern[str], float]] = [
    (EntityType.PART, re.compile(r"\b([A-Z]{2,4}-\d{3,5}(?:-[A-Z0-9]{1,3})?)\b"), 0.95),
    (EntityType.LOCATION, re.compile(r"(?i)\b(bay\s?\d+|line\s?\d+|cell\s?\d+|zone\s?[a-z0-9]+|"
                                     r"console\s?\d+|mast\s?\w*)\b"), 0.9),
    (EntityType.ASSET, re.compile(r"(?i)\b(conveyor|gantry|spindle|drive|bearing|pump|valve|"
                                  r"servo|relay|interlock|compressor|actuator)\b"), 0.8),
    (EntityType.METRIC, re.compile(r"(?i)\b(pressure|vibration|temperature|torque|current|"
                                   r"voltage|flow|speed|humidity|tension)\b"), 0.75),
    (EntityType.PERSON, re.compile(r"(?i)\b(operator\s+(?:id\s+)?[a-z]?\d{2,})\b"), 0.9),
    (EntityType.THRESHOLD, re.compile(r"\b(\d+(?:\.\d+)?)\s*(mm/s|bar|nm|rpm|°?c|hz|kw|v|a)\b"), 0.85),
]

RELATION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("exceeded", re.compile(r"(?i)\b(crossed|exceeded|above|over|climbed to|rose to)\b")),
    ("below", re.compile(r"(?i)\b(below|under|dropped to|fell to|read)\b")),
    ("located_in", re.compile(r"(?i)\b(in|on|at)\b")),
    ("replaced", re.compile(r"(?i)\b(replaced|swapped|changed|fitted)\b")),
    ("caused", re.compile(r"(?i)\b(caused|triggered|fired|tripped|led to|resulted in)\b")),
    ("requires", re.compile(r"(?i)\b(requires|needs|must|then|before|after)\b")),
    ("acknowledged", re.compile(r"(?i)\b(acknowledged|confirmed|signed off|approved)\b")),
    ("measured", re.compile(r"(?i)\b(measured|read|logged|recorded|reported)\b")),
]

_NORMALISE = re.compile(r"\s+")


def canonical(name: str) -> str:
    return _NORMALISE.sub(" ", name.strip().lower())


@dataclass
class Entity:
    entity_id: str
    name: str
    type: EntityType
    aliases: set[str] = field(default_factory=set)
    mentions: list[str] = field(default_factory=list)      # point ids
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    confidence: float = 0.8

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.entity_id, "name": self.name, "type": self.type.value,
                "aliases": sorted(self.aliases), "mentions": len(self.mentions),
                "first_seen": self.first_seen, "last_seen": self.last_seen,
                "confidence": round(self.confidence, 3)}


@dataclass
class Fact:
    """One edge, with both time axes."""
    fact_id: str
    subject: str
    predicate: str
    object: str
    valid_from: float
    valid_to: float = INFINITY
    recorded_at: float = field(default_factory=time.time)
    retracted_at: float = INFINITY
    confidence: float = 0.8
    provenance: list[str] = field(default_factory=list)
    superseded_by: str | None = None

    def valid_at(self, when: float) -> bool:
        return self.valid_from <= when < self.valid_to

    def believed_at(self, when: float) -> bool:
        return self.recorded_at <= when < self.retracted_at

    def live(self, valid_time: float | None = None, as_of: float | None = None) -> bool:
        """Is this fact both true and believed at the given times?

        The clock is read only when neither time is supplied. It used to be
        read unconditionally, which cost a `time.time()` per incident fact per
        hop — measured at 117,504 calls to serve 192 queries, for an answer
        both callers already had.
        """
        if valid_time is None or as_of is None:
            now = time.time()
            valid_time = now if valid_time is None else valid_time
            as_of = now if as_of is None else as_of
        return self.valid_at(valid_time) and self.believed_at(as_of)

    @property
    def expires_at(self) -> float:
        """When this fact stops being live, if nothing changes."""
        return min(self.valid_to, self.retracted_at)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.fact_id, "subject": self.subject, "predicate": self.predicate,
            "object": self.object, "confidence": round(self.confidence, 3),
            "valid_from": self.valid_from,
            "valid_to": None if self.valid_to == INFINITY else self.valid_to,
            "recorded_at": self.recorded_at,
            "retracted_at": None if self.retracted_at == INFINITY else self.retracted_at,
            "provenance": self.provenance, "superseded_by": self.superseded_by,
        }


@dataclass
class Path:
    nodes: list[str]
    edges: list[Fact]

    @property
    def hops(self) -> int:
        return len(self.edges)

    @property
    def confidence(self) -> float:
        score = 1.0
        for edge in self.edges:
            score *= edge.confidence
        return round(score, 4)

    def as_dict(self) -> dict[str, Any]:
        return {"nodes": self.nodes, "hops": self.hops, "confidence": self.confidence,
                "edges": [f"{e.subject} --{e.predicate}--> {e.object}" for e in self.edges]}


class KnowledgeGraph:
    MAX_HOPS = 4

    def __init__(self) -> None:
        self.entities: dict[str, Entity] = {}
        self.facts: dict[str, Fact] = {}
        self.out: dict[str, list[str]] = defaultdict(list)
        self.into: dict[str, list[str]] = defaultdict(list)
        self.by_point: dict[str, list[str]] = defaultdict(list)     # point -> fact ids
        self.extractions = 0
        self.retractions = 0
        self._sequence = 0
        # Materialised live view, rebuilt when the graph changes or when a
        # fact's validity window closes. See `_rebuild_adjacency`.
        self._version = 0
        self._adjacency: dict[str, list[tuple[str, float]]] | None = None
        self._adjacency_version = -1
        self._adjacency_expiry = 0.0
        self._adjacency_builds = 0

    # -- extraction -------------------------------------------------------

    def extract(self, point_id: str, text: str, when: float | None = None,
                collection: str = "episodic") -> tuple[list[Entity], list[Fact]]:
        """Pull entities and relations out of one memory."""
        when = when or time.time()
        self.extractions += 1
        found: list[Entity] = []
        spans: list[tuple[int, int, Entity]] = []

        for entity_type, pattern, confidence in ENTITY_PATTERNS:
            for match in pattern.finditer(text):
                raw = match.group(0)
                name = canonical(raw)
                entity_id = f"{entity_type.value}:{name}"
                entity = self.entities.get(entity_id)
                if entity is None:
                    entity = Entity(entity_id, name, entity_type, confidence=confidence)
                    self.entities[entity_id] = entity
                entity.aliases.add(raw)
                entity.last_seen = when
                if point_id not in entity.mentions:
                    entity.mentions.append(point_id)
                spans.append((match.start(), match.end(), entity))
                found.append(entity)

        spans.sort(key=lambda row: row[0])
        facts: list[Fact] = []
        for index in range(len(spans) - 1):
            left_end, left = spans[index][1], spans[index][2]
            right_start, right = spans[index + 1][0], spans[index + 1][2]
            if left.entity_id == right.entity_id:
                continue
            between = text[left_end:right_start]
            if len(between) > 80:
                continue                    # too far apart to be one relation
            predicate = self._predicate(between, left, right)
            if predicate is None:
                continue
            facts.append(self.assert_fact(
                left.entity_id, predicate, right.entity_id, valid_from=when,
                confidence=min(left.confidence, right.confidence) * 0.92,
                provenance=[point_id],
            ))

        # the memory itself is a node, so retrieval can walk back to text
        for entity in found:
            self.by_point[point_id].append(entity.entity_id)
        return found, facts

    @staticmethod
    def _predicate(between: str, left: Entity, right: Entity) -> str | None:
        for name, pattern in RELATION_PATTERNS:
            if pattern.search(between):
                return name
        if right.type is EntityType.THRESHOLD and left.type is EntityType.METRIC:
            return "measured"
        if left.type in {EntityType.ASSET, EntityType.PART} and right.type is EntityType.LOCATION:
            return "located_in"
        if left.type is EntityType.LOCATION and right.type in {EntityType.ASSET, EntityType.PART}:
            return "contains"
        return None

    # -- writes -----------------------------------------------------------

    def assert_fact(self, subject: str, predicate: str, object_: str, *,
                    valid_from: float | None = None, valid_to: float = INFINITY,
                    confidence: float = 0.8, provenance: Iterable[str] = ()) -> Fact:
        """Assert an edge, merging into an identical live one.

        The same relation observed in a second memory is corroboration, not a
        new fact. Duplicating it would inflate every graph statistic and let
        repetition masquerade as evidence, so provenance is merged and
        confidence is raised toward — never past — certainty.
        """
        provenance = list(provenance)
        existing = self._find_live(subject, predicate, object_)
        if existing is not None:
            for point_id in provenance:
                if point_id not in existing.provenance:
                    existing.provenance.append(point_id)
                    self.by_point[point_id].append(existing.fact_id)
            corroborations = max(len(existing.provenance) - 1, 0)
            existing.confidence = min(0.99, existing.confidence + 0.03 * corroborations)
            existing.valid_from = min(existing.valid_from, valid_from or existing.valid_from)
            return existing

        self._sequence += 1
        fact_id = f"f{self._sequence:08d}"
        fact = Fact(fact_id=fact_id, subject=subject, predicate=predicate, object=object_,
                    valid_from=valid_from or time.time(), valid_to=valid_to,
                    confidence=confidence, provenance=list(provenance))
        self.facts[fact_id] = fact
        self._version += 1
        self.out[subject].append(fact_id)
        self.into[object_].append(fact_id)
        for point_id in fact.provenance:
            self.by_point[point_id].append(fact_id)
        return fact

    def _find_live(self, subject: str, predicate: str, object_: str) -> Fact | None:
        for fact_id in self.out.get(subject, []):
            fact = self.facts.get(fact_id)
            if (fact is not None and fact.predicate == predicate and fact.object == object_
                    and fact.retracted_at == INFINITY):
                return fact
        return None

    def _touch(self) -> None:
        """Mark the live view stale. Cheap, and the only thing a mutation owes."""
        self._version += 1

    def retract(self, fact_id: str, superseded_by: str | None = None,
                at: float | None = None) -> bool:
        """Stop believing a fact without deleting it.

        A deleted fact cannot be audited and cannot answer "when did we stop
        believing this" — which is exactly what an incident review asks.
        """
        fact = self.facts.get(fact_id)
        if fact is None or fact.retracted_at != INFINITY:
            return False
        fact.retracted_at = at or time.time()
        fact.superseded_by = superseded_by
        self.retractions += 1
        self._touch()
        return True

    def supersede(self, old_fact_id: str, new_fact: Fact) -> bool:
        return self.retract(old_fact_id, superseded_by=new_fact.fact_id)

    def forget_point(self, point_id: str) -> int:
        """A deleted memory retracts the facts it was the sole evidence for."""
        removed = 0
        for fact_id in list(self.by_point.get(point_id, [])):
            fact = self.facts.get(fact_id)
            if fact is None:
                continue
            fact.provenance = [p for p in fact.provenance if p != point_id]
            if not fact.provenance and self.retract(fact_id):
                removed += 1
        self.by_point.pop(point_id, None)
        return removed

    # -- queries ----------------------------------------------------------

    def live_facts(self, valid_time: float | None = None, as_of: float | None = None) -> list[Fact]:
        return [f for f in self.facts.values() if f.live(valid_time, as_of)]

    def _rebuild_adjacency(self, now: float) -> None:
        """Materialise the live graph once, instead of re-deriving it per hop.

        Spreading activation walks the same neighbourhoods over and over, and
        each visit was filtering every incident fact for liveness and
        de-duplicating the result into a fresh dict. Profiling 192 queries put
        59,136 calls through that path — 29% of all query CPU.

        So the live view is built once and reused until something changes it.
        Two things can: a mutation, which bumps a version, and the passage of
        time, because a fact with a `valid_to` in the future is live now and
        will not be later. The earliest such moment is kept as an expiry, so
        the cache is correct rather than merely fast.
        """
        adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
        expiry = INFINITY
        for fact in self.facts.values():
            if not fact.live(now, now):
                continue
            adjacency[fact.subject].append((fact.object, fact.confidence))
            adjacency[fact.object].append((fact.subject, fact.confidence))
            expiry = min(expiry, fact.expires_at)
        self._adjacency = adjacency
        self._adjacency_expiry = expiry
        self._adjacency_version = self._version
        self._adjacency_builds += 1

    def live_adjacency(self, now: float | None = None) -> dict[str, list[tuple[str, float]]]:
        """Neighbours of every entity, as the graph stands right now."""
        now = now if now is not None else time.time()
        if (self._adjacency is None or self._adjacency_version != self._version
                or now >= self._adjacency_expiry):
            self._rebuild_adjacency(now)
        return self._adjacency

    def neighbours(self, entity_id: str, valid_time: float | None = None,
                   as_of: float | None = None, direction: str = "both") -> list[Fact]:
        fact_ids: list[str] = []
        if direction in {"out", "both"}:
            fact_ids.extend(self.out.get(entity_id, []))
        if direction in {"in", "both"}:
            fact_ids.extend(self.into.get(entity_id, []))
        return [self.facts[f] for f in dict.fromkeys(fact_ids)
                if self.facts[f].live(valid_time, as_of)]

    def paths(self, source: str, target: str, max_hops: int | None = None,
              valid_time: float | None = None, as_of: float | None = None) -> list[Path]:
        """Bounded BFS over the graph as it was believed at `as_of`."""
        max_hops = min(max_hops or self.MAX_HOPS, self.MAX_HOPS)
        if source not in self.entities or target not in self.entities:
            return []
        found: list[Path] = []
        queue: deque[tuple[str, list[str], list[Fact]]] = deque([(source, [source], [])])
        seen: set[tuple[str, int]] = {(source, 0)}
        while queue:
            node, nodes, edges = queue.popleft()
            if len(edges) >= max_hops:
                continue
            # Walk edges in both directions. Relations are asserted in whatever
            # order the sentence happened to put them in ("bay 3 contains
            # conveyor" vs "bearing located_in bay 3"), so a direction-only walk
            # answers structurally identical questions differently depending on
            # phrasing — which is not a property anyone can reason about.
            for fact in self.neighbours(node, valid_time, as_of, direction="both"):
                nxt = fact.object if fact.subject == node else fact.subject
                if nxt in nodes:
                    continue
                if nxt == target:
                    found.append(Path([*nodes, nxt], [*edges, fact]))
                    continue
                key = (nxt, len(edges) + 1)
                if key in seen:
                    continue
                seen.add(key)
                queue.append((nxt, [*nodes, nxt], [*edges, fact]))
        found.sort(key=lambda p: (p.hops, -p.confidence))
        return found[:8]

    def as_of(self, when: float, valid_time: float | None = None) -> dict[str, Any]:
        """What this node believed at a point in time — the audit question."""
        facts = [f for f in self.facts.values() if f.live(valid_time or when, when)]
        return {
            "as_of": when, "valid_time": valid_time or when,
            "facts": len(facts),
            "entities": len({e for f in facts for e in (f.subject, f.object)}),
            "sample": [f.as_dict() for f in facts[:20]],
        }

    def diff_beliefs(self, earlier: float, later: float) -> dict[str, Any]:
        """What changed between two moments — learned, retracted, still held."""
        before = {f.fact_id for f in self.facts.values() if f.believed_at(earlier)}
        after = {f.fact_id for f in self.facts.values() if f.believed_at(later)}
        return {
            "learned": [self.facts[f].as_dict() for f in sorted(after - before)][:20],
            "retracted": [self.facts[f].as_dict() for f in sorted(before - after)][:20],
            "learned_count": len(after - before), "retracted_count": len(before - after),
            "stable_count": len(before & after),
        }

    def spreading_activation(self, seeds: list[str], hops: int = 2, decay: float = 0.55,
                             as_of: float | None = None) -> dict[str, float]:
        """Entity relevance by graph diffusion — the retrieval boost signal."""
        activation: dict[str, float] = {seed: 1.0 for seed in seeds if seed in self.entities}
        frontier = dict(activation)
        # Time travel takes the slow path deliberately: a cache of "now" must
        # never answer a question about what was believed at some other time.
        adjacency = self.live_adjacency() if as_of is None else None
        for _ in range(hops):
            nxt: dict[str, float] = defaultdict(float)
            for node, energy in frontier.items():
                if adjacency is not None:
                    for other, confidence in adjacency.get(node, ()):
                        nxt[other] += energy * decay * confidence
                    continue
                for fact in self.neighbours(node, as_of=as_of):
                    other = fact.object if fact.subject == node else fact.subject
                    nxt[other] += energy * decay * fact.confidence
            frontier = {}
            for node, energy in nxt.items():
                if energy > 0.02 and energy > activation.get(node, 0.0):
                    activation[node] = energy
                    frontier[node] = energy
            if not frontier:
                break
        return activation

    def points_for_entities(self, entity_ids: Iterable[str]) -> dict[str, float]:
        """Map activated entities back to the memories that mention them."""
        scores: dict[str, float] = defaultdict(float)
        for entity_id in entity_ids:
            entity = self.entities.get(entity_id)
            if entity is None:
                continue
            for point_id in entity.mentions:
                scores[point_id] += 1.0
        return dict(scores)

    def entities_in(self, text: str) -> list[str]:
        ids: list[str] = []
        for entity_type, pattern, _ in ENTITY_PATTERNS:
            for match in pattern.finditer(text):
                entity_id = f"{entity_type.value}:{canonical(match.group(0))}"
                if entity_id in self.entities:
                    ids.append(entity_id)
        return list(dict.fromkeys(ids))

    # -- reporting --------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        by_type: dict[str, int] = defaultdict(int)
        for entity in self.entities.values():
            by_type[entity.type.value] += 1
        by_predicate: dict[str, int] = defaultdict(int)
        live = 0
        for fact in self.facts.values():
            by_predicate[fact.predicate] += 1
            live += int(fact.live())
        return {
            "entities": len(self.entities), "by_type": dict(by_type),
            "facts": len(self.facts), "live_facts": live,
            "by_predicate": dict(sorted(by_predicate.items(), key=lambda kv: -kv[1])[:8]),
            "retractions": self.retractions, "extractions": self.extractions,
        }
