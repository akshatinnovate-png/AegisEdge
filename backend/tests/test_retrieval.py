"""Retrieval: hybrid beats either half, ranking terms, cache, agent."""
from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_sparse_recall_finds_a_rare_token_dense_misses(node):
    await node.remember("Replaced bearing housing part BX-7741-Q on the bay 3 conveyor")
    for i in range(12):
        await node.remember(f"routine shift note {i} about the line running normally")
    result = await node.pipeline.search("BX-7741-Q", k=3, mode="sparse")
    assert result.results
    assert "BX-7741-Q" in result.results[0]["text"]


@pytest.mark.asyncio
async def test_hybrid_reports_which_space_matched(node):
    await node.remember("Coolant pressure below 1.8 bar is a hard stop condition", collection="semantic")
    result = await node.pipeline.search("coolant pressure hard stop", k=3)
    assert result.results
    assert set(result.results[0]["matched_by"]) == {"dense", "sparse"}
    assert result.stages["embed_ms"] >= 0 and "rerank_ms" in result.stages


@pytest.mark.asyncio
async def test_superseded_memories_sink_in_the_ranking(node):
    old = await node.remember("The torque limit on line 2 is 30 Nm", collection="semantic")
    new = await node.remember("The torque limit on line 2 is 42 Nm", collection="semantic")
    node.store.supersede(old.id, new.id, reason="test")
    node.pipeline.cache.invalidate()
    result = await node.pipeline.search("torque limit line 2", k=2)
    ids = [r["id"] for r in result.results]
    assert ids.index(new.id) < ids.index(old.id)


@pytest.mark.asyncio
async def test_semantic_cache_serves_a_near_identical_query(node):
    await node.remember("Gantry re-homed after the interlock released", collection="episodic")
    await node.pipeline.search("gantry re-homed", k=3)
    again = await node.pipeline.search("gantry re-homed", k=3)
    assert again.cached
    assert node.pipeline.cache.snapshot()["hits"] >= 1


@pytest.mark.asyncio
async def test_a_write_invalidates_the_cache(node):
    await node.remember("Line 1 is idle", collection="episodic")
    await node.pipeline.search("line 1 status", k=2)
    await node.remember("Line 1 resumed production at 06:10", collection="episodic")
    fresh = await node.pipeline.search("line 1 status", k=2)
    assert not fresh.cached


@pytest.mark.asyncio
async def test_escalation_is_declined_while_offline(node):
    node.triton.url = "grpc://triton.internal:8001"
    node.oracle.forced_offline = True
    await node.oracle.probe_once()
    await node.remember("why did the conveyor stop overnight", collection="episodic")
    result = await node.pipeline.search("why did the conveyor stop", k=3)
    assert not result.escalated
    assert "offline" in result.escalation["reason"]


@pytest.mark.asyncio
async def test_restricted_results_block_escalation(node):
    node.triton.url = "grpc://triton.internal:8001"
    await node.oracle.probe_once()
    await node.remember("Operator 4471 jo@plant.io explained why the line stopped")
    result = await node.pipeline.search("why did operator 4471 stop the line", k=1)
    assert not result.escalated
    assert result.escalation["checks"]["may_egress"] is False


@pytest.mark.asyncio
async def test_agent_answers_with_citations_and_a_trace(node):
    await node.remember("Recovery: isolate the drive, purge the line, re-home the gantry",
                        collection="procedural")
    answer = await node.agent.answer("how do i recover the drive")
    assert answer.citations
    assert [s["step"] for s in answer.trace] == ["plan", "retrieve", "verify", "answer"]
    assert answer.confidence > 0


@pytest.mark.asyncio
async def test_agent_flags_a_contradiction(node):
    await node.remember("Coolant pressure below 1.8 bar is a hard stop", collection="semantic")
    await node.remember("The 1.8 bar hard stop was lifted for bay 3 this week", collection="semantic")
    answer = await node.agent.answer("is 1.8 bar a hard stop")
    assert answer.contradictions


# -- scope: what the caller asked vs what the node guessed --------------------

@pytest.mark.asyncio
async def test_semantic_cache_is_keyed_by_scope_not_just_tenant(node):
    """A cached answer must not escape the question it was the answer to.

    The cache was namespaced by tenant after a cross-tenant leak, but not by
    collection, mode or k. So a query run once over all collections was then
    served verbatim for `collection="procedural"` — five episodic hits from a
    collection holding nothing — and every scoped query after it was wrong in
    the same way.
    """
    for i in range(20):
        await node.remember(f"bay {i % 4} conveyor vibration crossed {i / 10:.1f} mm/s")

    everywhere = await node.pipeline.search("conveyor", k=5, collection="*")
    assert everywhere.results

    for empty in ("procedural", "semantic"):
        scoped = await node.pipeline.search("conveyor", k=5, collection=empty)
        assert scoped.results == [], f"{empty} is empty but returned hits"

    # ...and the cache must still work within a single scope.
    again = await node.pipeline.search("conveyor", k=5, collection="*")
    assert len(again.results) == len(everywhere.results)


@pytest.mark.asyncio
async def test_an_inferred_narrowing_never_empties_the_results(node):
    """Query understanding may guess; it may not silently answer nothing.

    "conveyor vibration night shift" reads as sensor intent, so the pipeline
    narrowed to the sensor collection. With those memories stored as episodic
    the hard filter returned zero hits — while the single word "conveyor"
    returned five, because it inferred nothing at all.
    """
    for i in range(20):
        await node.remember(
            f"bay {i % 4} conveyor vibration crossed {i / 10:.1f} mm/s during the night shift")

    result = await node.pipeline.search("conveyor vibration night shift", k=5)
    assert result.results, "an inferred filter emptied the result set"

    relaxed = result.as_dict()["understanding"].get("inference_relaxed")
    assert relaxed is not None
    assert relaxed["dropped"]
    assert relaxed["recovered_hits"] == len(result.results)


@pytest.mark.asyncio
async def test_an_explicit_filter_is_obeyed_even_when_it_matches_nothing(node):
    """Only our own guess is ever backed out. The caller's question is theirs."""
    for i in range(20):
        await node.remember(f"bay {i % 4} conveyor vibration crossed {i / 10:.1f} mm/s")

    assert (await node.pipeline.search("conveyor", k=5, collection="procedural")).results == []
    explicit = await node.pipeline.search(
        "conveyor", k=5, filters={"collection": "procedural"})
    assert explicit.results == []
    assert "inference_relaxed" not in explicit.as_dict().get("understanding", {})


@pytest.mark.asyncio
async def test_tenant_isolation_is_never_relaxed(node):
    """An empty visible set is a boundary, not a guess that went wrong."""
    for i in range(10):
        await node.remember(f"bay {i} conveyor vibration crossed {i / 10:.1f} mm/s")
    ghost = await node.pipeline.search("conveyor vibration night shift", k=5,
                                       tenant_id="nobody-owns-this")
    assert ghost.results == []
    assert "inference_relaxed" not in ghost.as_dict().get("understanding", {})
