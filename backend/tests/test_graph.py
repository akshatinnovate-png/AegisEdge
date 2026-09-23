"""Knowledge graph: extraction, structure, bitemporal beliefs, reasoning."""
from __future__ import annotations

import time

import pytest

from aegis.memory.graph import EntityType, KnowledgeGraph

CORPUS = [
    ("p1", "Bay 3 conveyor vibration crossed 4.2 mm/s at 02:14"),
    ("p2", "Bearing BX-7741-Q on the bay 3 conveyor was replaced during the night shift"),
    ("p3", "Coolant pressure in line 2 read 1.74 bar before the interlock fired"),
    ("p4", "Operator 4471 acknowledged the torque alarm on line 2"),
    ("p5", "The bay 3 conveyor feeds line 2 directly"),
]


@pytest.fixture
def graph() -> KnowledgeGraph:
    kg = KnowledgeGraph()
    base = time.time() - 86400
    for index, (point_id, text) in enumerate(CORPUS):
        kg.extract(point_id, text, base + index * 3600)
    return kg


def test_extraction_finds_domain_entities(graph):
    assert "part:bx-7741-q" in graph.entities
    assert "location:bay 3" in graph.entities
    assert graph.entities["part:bx-7741-q"].type is EntityType.PART
    assert graph.entities["person:operator 4471"].type is EntityType.PERSON


def test_identical_relations_are_merged_not_duplicated(graph):
    before = len(graph.facts)
    graph.extract("p6", "Bay 3 conveyor vibration crossed 4.2 mm/s again")
    duplicates = [f for f in graph.facts.values()
                  if f.subject == "location:bay 3" and f.object == "asset:conveyor"]
    assert len(duplicates) == 1
    assert len(graph.facts) - before <= 1
    assert len(duplicates[0].provenance) >= 2          # corroboration, not a new fact


def test_corroboration_raises_confidence_without_reaching_certainty(graph):
    fact = next(f for f in graph.facts.values() if f.subject == "location:bay 3")
    start = fact.confidence
    for i in range(10):
        graph.extract(f"extra{i}", "Bay 3 conveyor vibration crossed 4.2 mm/s")
    assert fact.confidence > start
    assert fact.confidence <= 0.99


def test_multi_hop_paths_traverse_edges_in_either_direction(graph):
    paths = graph.paths("part:bx-7741-q", "location:line 2", max_hops=3)
    assert paths
    best = paths[0]
    assert best.nodes[0] == "part:bx-7741-q" and best.nodes[-1] == "location:line 2"
    assert 0 < best.confidence <= 1.0


def test_paths_are_bounded(graph):
    assert graph.paths("part:bx-7741-q", "location:line 2", max_hops=99)[0].hops <= graph.MAX_HOPS


def test_retraction_preserves_the_record(graph):
    fact = next(iter(graph.facts.values()))
    assert graph.retract(fact.fact_id)
    assert not graph.retract(fact.fact_id)             # idempotent
    assert fact.fact_id in graph.facts                 # still auditable
    assert fact.as_dict()["retracted_at"] is not None
    assert fact not in graph.live_facts()


def test_bitemporal_as_of_answers_what_we_believed_then(graph):
    believed_then = time.time()
    time.sleep(0.02)
    fact = next(f for f in graph.facts.values() if f.live())
    graph.retract(fact.fact_id)
    now = time.time()

    assert graph.as_of(believed_then)["facts"] == graph.as_of(now)["facts"] + 1
    diff = graph.diff_beliefs(believed_then, now)
    assert diff["retracted_count"] == 1
    assert diff["learned_count"] == 0


def test_valid_time_and_transaction_time_are_independent(graph):
    past = time.time() - 86400 * 30
    fact = graph.assert_fact("asset:pump", "located_in", "location:bay 3",
                             valid_from=past, valid_to=past + 3600)
    # believed now, but only true for an hour a month ago
    assert not fact.live()
    assert fact.live(valid_time=past + 60)
    assert fact.believed_at(time.time())


def test_spreading_activation_reaches_indirect_neighbours(graph):
    activation = graph.spreading_activation(["part:bx-7741-q"], hops=2)
    assert activation["part:bx-7741-q"] == 1.0
    assert len(activation) > 1
    assert all(0 < value <= 1.0 for value in activation.values())


def test_activation_maps_back_to_memories(graph):
    activation = graph.spreading_activation(["metric:pressure"], hops=2)
    points = graph.points_for_entities(activation)
    assert "p3" in points


def test_deleting_a_memory_retracts_facts_it_alone_supported(graph):
    live_before = len(graph.live_facts())
    graph.forget_point("p3")
    assert len(graph.live_facts()) < live_before


@pytest.mark.asyncio
async def test_ingest_populates_the_graph(node):
    await node.remember("Bearing BX-9001-A on the bay 7 conveyor exceeded 5.1 mm/s",
                        collection="sensor")
    assert "part:bx-9001-a" in node.graph.entities
    assert node.graph.snapshot()["facts"] > 0


@pytest.mark.asyncio
async def test_graph_boosts_structurally_related_memories(node):
    await node.remember("Bearing BX-7741-Q was fitted to the bay 3 conveyor", collection="episodic")
    await node.remember("Bay 3 conveyor vibration crossed 4.2 mm/s", collection="sensor")
    result = await node.pipeline.search("vibration on the conveyor", k=5)
    assert result.graph_context.get("seed_entities")
    assert result.graph_context.get("boosted_points", 0) > 0
