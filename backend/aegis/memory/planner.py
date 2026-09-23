"""Query planner.

Given a filter and a vector, there are three ways to run the query and the
wrong one is 100x slower:

* **pre-filter** — resolve the filter, then brute-force the matching subset.
  Wins when the filter is selective: 200 candidates is nothing to scan.
* **post-filter** — run ANN over everything and discard non-matches. Wins when
  the filter is loose, but silently loses recall if too few survive, so the
  plan must over-fetch by the inverse selectivity.
* **full scan** — exact over everything. Only for small collections.

The planner estimates selectivity from payload statistics, computes the cost
of each, and explains the choice. Every plan is attached to the result so a
slow query can be diagnosed instead of guessed at.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .filters import Filter, PayloadIndex


class PlanKind(str, Enum):
    PRE_FILTER = "pre_filter"
    POST_FILTER = "post_filter"
    FULL_SCAN = "full_scan"
    EMPTY = "empty"


@dataclass
class QueryPlan:
    kind: PlanKind
    selectivity: float
    estimated_matches: int
    fetch_k: int
    cost_estimate: float
    reason: str
    allow: set[str] | None = None
    alternatives: dict[str, float] = field(default_factory=dict)
    planned_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {"plan": self.kind.value, "selectivity": round(self.selectivity, 5),
                "estimated_matches": self.estimated_matches, "fetch_k": self.fetch_k,
                "cost": round(self.cost_estimate, 2), "reason": self.reason,
                "alternatives": {k: round(v, 2) for k, v in self.alternatives.items()}}


class QueryPlanner:
    # relative costs, in units of "one full-precision distance evaluation"
    COST_SCAN_POINT = 1.0
    COST_ANN_POINT = 0.06          # ANN touches a small fraction of the corpus
    COST_FILTER_CHECK = 0.15
    COST_INDEX_LOOKUP = 0.02

    def __init__(self, payload_index: PayloadIndex) -> None:
        self.index = payload_index
        self.plans = 0
        self.by_kind: dict[str, int] = {}

    def plan(self, spec: Filter, corpus: int, k: int, ann_available: bool) -> QueryPlan:
        self.plans += 1
        if spec.is_empty():
            plan = QueryPlan(
                kind=PlanKind.POST_FILTER if ann_available else PlanKind.FULL_SCAN,
                selectivity=1.0, estimated_matches=corpus, fetch_k=k,
                cost_estimate=corpus * (self.COST_ANN_POINT if ann_available else self.COST_SCAN_POINT),
                reason="no filter — straight vector search",
            )
            return self._record(plan)

        selectivity = self.index.estimate(spec)
        matches = max(1, int(round(selectivity * corpus)))

        resolved = self.index.resolve_filter(spec)
        exact = resolved is not None
        if exact:
            matches = len(resolved)
            selectivity = matches / corpus if corpus else 0.0

        if exact and not resolved:
            return self._record(QueryPlan(
                kind=PlanKind.EMPTY, selectivity=0.0, estimated_matches=0, fetch_k=0,
                cost_estimate=self.COST_INDEX_LOOKUP,
                reason="filter provably matches nothing — no vector work at all",
                allow=set()))

        # over-fetch enough that the filter leaves k survivors
        post_fetch = min(corpus, max(k, int(k / max(selectivity, 1e-3))))
        cost_pre = matches * self.COST_SCAN_POINT + len(spec.must) * self.COST_INDEX_LOOKUP
        cost_post = (post_fetch * self.COST_ANN_POINT * (corpus / max(post_fetch, 1))
                     + post_fetch * self.COST_FILTER_CHECK) if ann_available else float("inf")
        cost_scan = corpus * (self.COST_SCAN_POINT + self.COST_FILTER_CHECK)

        options = {"pre_filter": cost_pre, "post_filter": cost_post, "full_scan": cost_scan}
        if not exact:
            options.pop("pre_filter")        # cannot pre-filter what we cannot resolve
        kind_name = min(options, key=options.get)

        if kind_name == "pre_filter":
            reason = (f"filter resolves to {matches} ids ({selectivity:.2%}) — "
                      f"scanning that subset exactly beats approximating the whole corpus")
            plan = QueryPlan(PlanKind.PRE_FILTER, selectivity, matches, k, cost_pre, reason,
                             allow=resolved, alternatives=options)
        elif kind_name == "post_filter":
            reason = (f"filter is loose ({selectivity:.1%}) — ANN first, over-fetching "
                      f"{post_fetch} to leave {k} after filtering")
            plan = QueryPlan(PlanKind.POST_FILTER, selectivity, matches, post_fetch, cost_post,
                             reason, alternatives=options)
        else:
            reason = f"corpus is small ({corpus}) — exact scan is cheapest and has no recall risk"
            plan = QueryPlan(PlanKind.FULL_SCAN, selectivity, matches, max(k, post_fetch),
                             cost_scan, reason, alternatives=options)
        return self._record(plan)

    def _record(self, plan: QueryPlan) -> QueryPlan:
        self.by_kind[plan.kind.value] = self.by_kind.get(plan.kind.value, 0) + 1
        return plan

    def snapshot(self) -> dict[str, Any]:
        return {"plans": self.plans, "by_kind": dict(self.by_kind), "index": self.index.snapshot()}
