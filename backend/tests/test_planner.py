"""Filters, payload statistics and the cost-based planner."""
from __future__ import annotations

import random

import pytest

from aegis.memory.filters import Condition, Filter, Op, PayloadIndex
from aegis.memory.planner import PlanKind, QueryPlanner


@pytest.fixture
def index() -> PayloadIndex:
    payload_index = PayloadIndex(["collection", "sensitivity", "ts"])
    random.seed(11)
    for i in range(1000):
        payload_index.index(f"p{i}", {
            "collection": random.choice(["episodic"] * 80 + ["sensor"] * 18 + ["procedural"] * 2),
            "sensitivity": random.choice(["internal"] * 95 + ["restricted"] * 5),
            "ts": float(i),
        })
    return payload_index


def test_shorthand_and_explicit_filters_agree():
    short = Filter.parse({"collection": "sensor", "ts": {"gte": 5.0}})
    explicit = Filter.parse({"must": [{"field": "collection", "op": "eq", "value": "sensor"},
                                      {"field": "ts", "op": "gte", "value": 5.0}]})
    payload = {"collection": "sensor", "ts": 9.0}
    assert short.matches(payload) and explicit.matches(payload)
    assert not short.matches({"collection": "sensor", "ts": 1.0})


def test_filters_never_raise_on_heterogeneous_payloads():
    spec = Filter.parse({"ts": {"gte": 5.0}})
    assert not spec.matches({"ts": "not-a-number"})           # comparison must not explode


def test_must_not_and_should_semantics():
    spec = Filter(should=[Condition("collection", Op.EQ, "sensor"),
                          Condition("collection", Op.EQ, "procedural")],
                  must_not=[Condition("sensitivity", Op.EQ, "restricted")])
    assert spec.matches({"collection": "sensor", "sensitivity": "internal"})
    assert not spec.matches({"collection": "sensor", "sensitivity": "restricted"})
    assert not spec.matches({"collection": "episodic", "sensitivity": "internal"})


def test_selectivity_tracks_reality(index: PayloadIndex):
    rare = index.selectivity(Condition("collection", Op.EQ, "procedural"))
    common = index.selectivity(Condition("collection", Op.EQ, "episodic"))
    assert rare < 0.08 < common
    ranged = index.selectivity(Condition("ts", Op.RANGE, (0.0, 99.0)))
    assert 0.05 < ranged < 0.15


def test_planner_prefilters_a_selective_query(index: PayloadIndex):
    plan = QueryPlanner(index).plan(Filter.parse({"collection": "procedural"}), 1000, 5, True)
    assert plan.kind is PlanKind.PRE_FILTER
    assert plan.allow is not None and len(plan.allow) == plan.estimated_matches


def test_planner_postfilters_a_loose_query(index: PayloadIndex):
    plan = QueryPlanner(index).plan(Filter.parse({"collection": "episodic"}), 1000, 5, True)
    assert plan.kind is PlanKind.POST_FILTER
    assert plan.fetch_k >= 5


def test_planner_short_circuits_an_impossible_filter(index: PayloadIndex):
    plan = QueryPlanner(index).plan(Filter.parse({"collection": "nope"}), 1000, 5, True)
    assert plan.kind is PlanKind.EMPTY
    assert plan.allow == set()
    assert plan.cost_estimate < 1.0                          # no vector work at all


def test_planner_explains_and_ranks_alternatives(index: PayloadIndex):
    plan = QueryPlanner(index).plan(Filter.parse({"sensitivity": "restricted"}), 1000, 5, True)
    assert plan.reason
    assert plan.alternatives
    assert plan.cost_estimate == min(plan.alternatives.values())


def test_dropping_a_point_updates_the_statistics(index: PayloadIndex):
    before = index.selectivity(Condition("collection", Op.EQ, "procedural"))
    for point_id in list(index.keyword["collection"].get("procedural", set()))[:5]:
        index.drop(point_id)
    assert index.selectivity(Condition("collection", Op.EQ, "procedural")) < before
