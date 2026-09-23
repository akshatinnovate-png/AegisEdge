"""Payload filters and the indexes that make them cheap.

A filtered vector search is two problems wearing one coat: find the matching
set, and find the nearest vectors. Which you do first decides whether the
query takes a millisecond or a second, and the right answer depends entirely
on how selective the filter is — which the index has to actually know.

So every indexed field keeps cardinality statistics, and the filter tree can
estimate its own selectivity before anything is executed.
"""
from __future__ import annotations

import bisect
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable


class Op(str, Enum):
    EQ = "eq"
    NE = "ne"
    IN = "in"
    LT = "lt"
    LTE = "lte"
    GT = "gt"
    GTE = "gte"
    RANGE = "range"
    EXISTS = "exists"
    CONTAINS = "contains"


@dataclass(slots=True)
class Condition:
    field: str
    op: Op
    value: Any = None

    def matches(self, payload: dict[str, Any]) -> bool:
        present = self.field in payload
        actual = payload.get(self.field)
        if self.op is Op.EXISTS:
            return present == bool(self.value if self.value is not None else True)
        if not present:
            return False
        try:
            if self.op is Op.EQ:
                return actual == self.value
            if self.op is Op.NE:
                return actual != self.value
            if self.op is Op.IN:
                return actual in self.value
            if self.op is Op.LT:
                return actual < self.value
            if self.op is Op.LTE:
                return actual <= self.value
            if self.op is Op.GT:
                return actual > self.value
            if self.op is Op.GTE:
                return actual >= self.value
            if self.op is Op.RANGE:
                low, high = self.value
                return low <= actual <= high
            if self.op is Op.CONTAINS:
                return self.value in actual
        except TypeError:
            return False                      # heterogeneous payloads must not raise
        return False

    def as_dict(self) -> dict[str, Any]:
        return {"field": self.field, "op": self.op.value, "value": self.value}


@dataclass
class Filter:
    """A boolean tree: must (AND) / should (OR) / must_not (NOT)."""
    must: list[Condition] = field(default_factory=list)
    should: list[Condition] = field(default_factory=list)
    must_not: list[Condition] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.must or self.should or self.must_not)

    def matches(self, payload: dict[str, Any]) -> bool:
        if any(not c.matches(payload) for c in self.must):
            return False
        if self.should and not any(c.matches(payload) for c in self.should):
            return False
        if any(c.matches(payload) for c in self.must_not):
            return False
        return True

    @staticmethod
    def parse(spec: dict[str, Any] | None) -> "Filter":
        """Accepts the shorthand `{"collection": "sensor", "ts": {"gte": 1.0}}`."""
        if not spec:
            return Filter()
        if any(key in spec for key in ("must", "should", "must_not")):
            def build(rows: Iterable[dict[str, Any]]) -> list[Condition]:
                return [Condition(r["field"], Op(r.get("op", "eq")), r.get("value")) for r in rows]
            return Filter(build(spec.get("must", [])), build(spec.get("should", [])),
                          build(spec.get("must_not", [])))
        must: list[Condition] = []
        for key, value in spec.items():
            if isinstance(value, dict):
                for op, operand in value.items():
                    must.append(Condition(key, Op(op), operand))
            elif isinstance(value, list):
                must.append(Condition(key, Op.IN, value))
            else:
                must.append(Condition(key, Op.EQ, value))
        return Filter(must=must)

    def as_dict(self) -> dict[str, Any]:
        return {"must": [c.as_dict() for c in self.must],
                "should": [c.as_dict() for c in self.should],
                "must_not": [c.as_dict() for c in self.must_not]}


class PayloadIndex:
    """Keyword postings + a sorted list per numeric field, with statistics.

    Keyword fields get exact posting lists. Numeric fields get a sorted array
    so a range is two binary searches instead of a scan. Both keep enough
    distribution detail for the planner to estimate selectivity without
    executing anything.
    """

    def __init__(self, fields: Iterable[str]) -> None:
        self.fields = set(fields)
        self.keyword: dict[str, dict[Any, set[str]]] = {f: {} for f in self.fields}
        self.numeric: dict[str, list[tuple[float, str]]] = {f: [] for f in self.fields}
        self.total = 0
        self.lookups = 0
        self.estimates = 0

    def index(self, point_id: str, payload: dict[str, Any]) -> None:
        self.drop(point_id)
        self.total += 1
        for field_name in self.fields:
            if field_name not in payload:
                continue
            value = payload[field_name]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                bisect.insort(self.numeric[field_name], (float(value), point_id))
            else:
                key = value if isinstance(value, (str, bool)) else str(value)
                self.keyword[field_name].setdefault(key, set()).add(point_id)

    def drop(self, point_id: str) -> None:
        removed = False
        for field_name in self.fields:
            for bucket in self.keyword[field_name].values():
                if point_id in bucket:
                    bucket.discard(point_id)
                    removed = True
            rows = self.numeric[field_name]
            for i, (_, pid) in enumerate(rows):
                if pid == point_id:
                    rows.pop(i)
                    removed = True
                    break
        if removed:
            self.total = max(0, self.total - 1)

    # -- resolution -------------------------------------------------------

    def resolve(self, condition: Condition) -> set[str] | None:
        """Exact id set for an indexed condition, or None if unindexed."""
        if condition.field not in self.fields:
            return None
        self.lookups += 1
        keyword = self.keyword[condition.field]
        rows = self.numeric[condition.field]

        if condition.op is Op.EQ and keyword:
            return set(keyword.get(condition.value, set()))
        if condition.op is Op.IN and keyword:
            out: set[str] = set()
            for value in condition.value or []:
                out |= keyword.get(value, set())
            return out
        if rows and condition.op in {Op.LT, Op.LTE, Op.GT, Op.GTE, Op.RANGE}:
            keys = [value for value, _ in rows]
            if condition.op is Op.RANGE:
                low, high = condition.value
            elif condition.op in {Op.LT, Op.LTE}:
                low, high = float("-inf"), condition.value
            else:
                low, high = condition.value, float("inf")
            start = bisect.bisect_left(keys, low)
            end = bisect.bisect_right(keys, high)
            window = rows[start:end]
            if condition.op is Op.LT:
                window = [(v, p) for v, p in window if v < condition.value]
            if condition.op is Op.GT:
                window = [(v, p) for v, p in window if v > condition.value]
            return {pid for _, pid in window}
        return None

    def selectivity(self, condition: Condition) -> float:
        """Estimated fraction of the corpus that matches, without executing."""
        self.estimates += 1
        if not self.total:
            return 1.0
        if condition.field not in self.fields:
            return 0.5                        # unknown field: assume a coin flip
        keyword = self.keyword[condition.field]
        rows = self.numeric[condition.field]
        if condition.op is Op.EQ and keyword:
            return len(keyword.get(condition.value, ())) / self.total
        if condition.op is Op.IN and keyword:
            return min(1.0, sum(len(keyword.get(v, ())) for v in condition.value or []) / self.total)
        if condition.op is Op.NE and keyword:
            return 1.0 - len(keyword.get(condition.value, ())) / self.total
        if rows and condition.op in {Op.LT, Op.LTE, Op.GT, Op.GTE, Op.RANGE}:
            keys = [value for value, _ in rows]
            if condition.op is Op.RANGE:
                low, high = condition.value
            elif condition.op in {Op.LT, Op.LTE}:
                low, high = float("-inf"), condition.value
            else:
                low, high = condition.value, float("inf")
            span = bisect.bisect_right(keys, high) - bisect.bisect_left(keys, low)
            return span / self.total
        if condition.op is Op.EXISTS:
            present = sum(len(b) for b in keyword.values()) + len(rows)
            return min(1.0, present / self.total)
        return 0.3

    def estimate(self, spec: Filter) -> float:
        """Combine condition selectivities (independence assumption, stated)."""
        estimate = 1.0
        for condition in spec.must:
            estimate *= max(self.selectivity(condition), 1e-4)
        if spec.should:
            miss = 1.0
            for condition in spec.should:
                miss *= (1.0 - self.selectivity(condition))
            estimate *= (1.0 - miss)
        for condition in spec.must_not:
            estimate *= max(1.0 - self.selectivity(condition), 1e-4)
        return min(1.0, max(estimate, 0.0))

    def resolve_filter(self, spec: Filter) -> set[str] | None:
        """Exact id set if every `must` is indexed, else None (planner falls back)."""
        if not spec.must or spec.should or spec.must_not:
            return None
        resolved: set[str] | None = None
        for condition in spec.must:
            ids = self.resolve(condition)
            if ids is None:
                return None
            resolved = ids if resolved is None else (resolved & ids)
            if not resolved:
                return set()
        return resolved

    def snapshot(self) -> dict[str, Any]:
        return {
            "fields": sorted(self.fields), "indexed_points": self.total,
            "lookups": self.lookups, "estimates": self.estimates,
            "cardinality": {f: len(self.keyword[f]) or len(self.numeric[f]) for f in sorted(self.fields)},
        }
