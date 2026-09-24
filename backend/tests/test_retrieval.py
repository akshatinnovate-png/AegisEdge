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


@pytest.mark.asyncio
async def test_the_cache_is_reached_even_when_understanding_infers_a_filter(node):
    """The cache was disabled for almost every query, and nothing said so.

    Lookup was guarded by `if not filters`, which reads as conservative. Query
    understanding infers a collection filter on most queries, so the guard was
    almost always true: measured over 128 queries with 64 repeats, the cache
    held 0 entries and had recorded 0 hits and 0 misses. It had never been
    asked a question, let alone answered one.
    """
    for i in range(40):
        await node.remember(f"bay {i % 5} conveyor vibration crossed {3 + i / 20:.2f} mm/s",
                            collection="sensor")
    query = "vibration on the bay 2 conveyor"
    first = await node.pipeline.search(query, k=5)
    assert first.cached is False

    second = await node.pipeline.search(query, k=5)
    assert second.cached is True, "a repeated question was not served from cache"
    assert node.pipeline.cache.snapshot()["exact_hits"] >= 1


@pytest.mark.asyncio
async def test_a_repeat_is_answered_without_running_the_encoder(node):
    """The exact layer sits in front of the embed, which is the whole point.

    Keying only on the query vector means a hit still pays for the encoder —
    the most expensive part of the query the cache exists to avoid.
    """
    for i in range(30):
        await node.remember(f"coolant pressure read {1.5 + i / 50:.2f} bar", collection="sensor")
    before = node.embedder.batcher.items
    query = "coolant pressure reading"
    await node.pipeline.search(query, k=5)
    after_first = node.embedder.batcher.items
    assert after_first > before                      # the first one embeds

    await node.pipeline.search(query, k=5)
    assert node.embedder.batcher.items == after_first, "a cached repeat still embedded"
    assert node.pipeline.snapshot()["answered_before_embedding"] >= 1


@pytest.mark.asyncio
async def test_a_write_invalidates_both_cache_layers(node):
    """A cached answer must never outlive the corpus it was drawn from."""
    for i in range(20):
        await node.remember(f"bay {i % 4} conveyor vibration {3 + i / 10:.1f} mm/s",
                            collection="sensor")
    query = "conveyor vibration"
    await node.pipeline.search(query, k=5)
    assert (await node.pipeline.search(query, k=5)).cached is True

    await node.remember("bay 9 conveyor vibration 9.9 mm/s", collection="sensor")
    assert (await node.pipeline.search(query, k=5)).cached is False


@pytest.mark.asyncio
async def test_the_cache_does_not_serve_one_scope_an_answer_from_another(node):
    for i in range(20):
        await node.remember(f"bay {i % 4} conveyor vibration {3 + i / 10:.1f} mm/s",
                            collection="sensor")
    query = "conveyor vibration"
    wide = await node.pipeline.search(query, k=5, collection="*")
    assert wide.results
    scoped = await node.pipeline.search(query, k=5, collection="procedural")
    assert scoped.results == []
    filtered = await node.pipeline.search(query, k=5, filters={"collection": "procedural"})
    assert filtered.results == []
