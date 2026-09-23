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
