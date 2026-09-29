"""Which engine answers a query, and why it changes its mind."""
from __future__ import annotations

import pytest

from aegis.retrieval.routing import PathRouter


def _choose(router, **kwargs):
    return router.choose(engine_available=True, server=False, degraded=False, **kwargs)


def test_a_store_without_a_native_path_never_routes_to_the_engine():
    router = PathRouter()
    decision = router.choose(engine_available=False, server=False, degraded=False)
    assert decision.path == "index" and "no Qdrant-native path" in decision.reason


def test_a_server_takes_the_engine_without_weighing_anything():
    router = PathRouter()
    for _ in range(50):
        router.observe("engine", 9_000.0)        # absurdly slow, and irrelevant
    decision = router.choose(engine_available=True, server=True, degraded=False)
    assert decision.path == "engine"


def test_the_engine_keeps_the_query_while_it_fits_the_recall_budget():
    router = PathRouter(objective_ms=150.0)
    for _ in range(router.WARMUP + 2):
        _choose(router)
        router.observe("engine", 8.0)
    decision = _choose(router)
    assert decision.path == "engine"
    assert "within the" in decision.reason


def test_an_engine_over_the_budget_loses_the_query_but_is_still_sampled():
    router = PathRouter(objective_ms=40.0)       # a 10 ms recall budget
    for _ in range(router.WARMUP + 2):
        _choose(router)
        router.observe("engine", 32.0)
        router.observe("index", 1.0)
    assert _choose(router).path == "index"
    # ...and the estimate is not allowed to freeze: one in SAMPLE_EVERY goes
    # back to the engine, so a machine that gets faster is noticed.
    paths = []
    for _ in range(router.SAMPLE_EVERY * 2):
        paths.append(_choose(router).path)
    assert "engine" in paths
    assert router.samples >= 1


def test_the_engine_keeps_the_query_when_the_local_index_is_no_better():
    router = PathRouter(objective_ms=40.0)
    for _ in range(router.WARMUP + 2):
        _choose(router)
        router.observe("engine", 30.0)
        router.observe("index", 45.0)            # slower still
    decision = _choose(router)
    assert decision.path == "engine" and "no better" in decision.reason


def test_a_degraded_node_takes_the_faster_path_because_that_is_the_point():
    router = PathRouter()
    for _ in range(router.WARMUP + 2):
        _choose(router)
        router.observe("engine", 2.0)            # comfortably within budget
    decision = router.choose(engine_available=True, server=False, degraded=True)
    assert decision.path == "index" and "degraded" in decision.reason


def test_the_policy_can_pin_either_path_so_the_bake_off_can_measure_both():
    assert PathRouter("engine").choose(engine_available=True, server=False,
                                       degraded=True).path == "engine"
    assert PathRouter("index").choose(engine_available=True, server=True,
                                      degraded=False).path == "index"


def test_the_snapshot_reports_both_estimates_and_the_last_decision():
    router = PathRouter()
    _choose(router)
    router.observe("engine", 5.0)
    router.observe("index", 1.0)
    snap = router.snapshot()
    assert snap["engine_p95_ms"] == 5.0 and snap["index_p95_ms"] == 1.0
    assert snap["queries"] == 1 and snap["last"]["path"] == "engine"
    assert snap["recall_budget_ms"] == pytest.approx(37.5)


@pytest.mark.asyncio
async def test_an_engine_answer_still_says_which_space_matched(node):
    """Fusion inside the engine must not cost the explanation.

    The engine fuses internally and returns one ordering, so per-space
    attribution would be lost — except that the two per-space orderings ride
    back in the same batch, which is one round trip, not three.
    """
    await node.remember("Coolant pressure below 1.8 bar is a hard stop",
                        collection="semantic")
    node.pipeline.router.policy = "engine"
    result = await node.pipeline.search("coolant pressure hard stop", k=3,
                                       collection="semantic")
    assert result.results
    assert result.plan["engine"]["route"]["path"] == "engine"
    assert result.plan["engine"]["queries_in_call"] == 3     # fused + both spaces
    # One round trip for one collection. A `*` search is one per collection,
    # which is still four stages each rather than four stages times four.
    assert result.plan["engine"]["calls"] == 1
    assert set(result.results[0]["matched_by"]) == {"dense", "sparse"}


@pytest.mark.asyncio
async def test_a_tenancy_allow_set_is_never_handed_to_the_engine(node):
    """An id set the planner resolved is honoured here, not approximated there.

    Qdrant filters on payload; it cannot be handed a set of ids to restrict to.
    Routing such a query to the engine would mean dropping the restriction, so
    the router is not even consulted.
    """
    node.pipeline.router.policy = "engine"
    node.tenants.create("acme")
    node.tenants.create("globex")
    await node.remember("acme torque limit is 42 Nm", collection="semantic", tenant_id="acme")
    await node.remember("globex torque limit is 11 Nm", collection="semantic", tenant_id="globex")
    result = await node.pipeline.search("torque limit", k=3, collection="semantic",
                                       tenant_id="acme")
    assert result.plan["engine"]["route"]["path"] == "index"
    assert all("globex" not in row["text"] for row in result.results)


def test_an_engine_that_answers_nothing_stops_getting_the_queries():
    """A warm-up counted in samples never ends if there are never any samples.

    An engine that declines every query — a server that is down, collections
    all on the old schema — produced no latency samples, so the router stayed
    in "measuring the engine" forever and paid for the attempt every time.
    """
    router = PathRouter()
    for _ in range(router.WARMUP + 2):
        decision = _choose(router)
        router.attempted()                       # asked, and it declined
    assert _choose(router).path == "index"
    assert "answered none of" in router.last.reason
    # ...but it is retried periodically, because a server can come back
    paths = []
    for _ in range(router.SAMPLE_EVERY * 2):
        paths.append(_choose(router).path)
        router.attempted()
    assert "engine" in paths


def test_a_slow_engine_and_a_silent_one_are_not_the_same_state():
    slow, silent = PathRouter(objective_ms=40.0), PathRouter(objective_ms=40.0)
    for _ in range(slow.WARMUP + 2):
        _choose(slow)
        slow.attempted()
        slow.observe("engine", 32.0)
        _choose(silent)
        silent.attempted()
    assert "exceeds the" in _choose(slow).reason          # measured, and too slow
    assert "answered none of" in _choose(silent).reason   # never measured at all
