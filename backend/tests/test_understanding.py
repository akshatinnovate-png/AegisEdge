"""Query understanding and late interaction."""
from __future__ import annotations

import pytest

from aegis.retrieval.query_understanding import BKTree, QueryUnderstanding, levenshtein

CORPUS = [
    "Coolant pressure below 1.8 bar for over 90 seconds is a hard stop condition",
    "Bay 3 conveyor vibration crossed 4.2mm/s matching the bearing signature",
    "Recovery procedure: isolate the drive, purge the line, re-home the gantry",
    "Operator switched line 2 to manual feed after the torque alarm",
    "Coolant pressure sensor readings are logged every shift in bay 3",
]


@pytest.fixture
def understanding() -> QueryUnderstanding:
    engine = QueryUnderstanding()
    for text in CORPUS * 2:
        engine.observe(text)
    return engine


def test_bktree_finds_matches_through_distant_branches():
    tree = BKTree()
    for word in ["coolant", "pressure", "gantry", "interlock", "vibration", "bearing"]:
        tree.add(word)
    assert ("pressure", 1) in tree.search("presure", 2)
    assert ("gantry", 1) in tree.search("gantery", 2)
    assert tree.search("zzzzzzz", 1) == []


def test_bounded_levenshtein_abandons_early():
    assert levenshtein("kitten", "sitting", cap=3) == 3
    assert levenshtein("abc", "xyzzyx", cap=2) == 3            # cap + 1 signals "too far"


def test_corpus_vocabulary_repairs_domain_typos(understanding):
    analysis = understanding.analyse("colent presure in bay3")
    assert analysis.corrections["colent"] == "coolant"
    assert analysis.corrections["presure"] == "pressure"


def test_expansion_comes_from_local_cooccurrence(understanding):
    analysis = understanding.analyse("coolant")
    assert analysis.expansions
    assert "pressure" in analysis.expansions or "bar" in analysis.expansions


def test_units_are_normalised(understanding):
    analysis = understanding.analyse("vibration 4.2mm/s on the conveyor")
    assert {"value": 4.2, "unit": "mm/s"} in analysis.units
    assert "4.2 mm/s" in analysis.normalized or "4 2 mm" in analysis.normalized


def test_relative_time_becomes_a_filter(understanding):
    analysis = understanding.analyse("sensor readings from the last 2 hours")
    assert "ts" in analysis.filters and "gte" in analysis.filters["ts"]
    assert analysis.filters["collection"] == "sensor"


def test_procedural_intent_routes_to_the_right_collection(understanding):
    assert understanding.analyse("how do i re-home the gantry").collection == "procedural"


@pytest.mark.asyncio
async def test_late_interaction_aligns_query_terms_to_document_terms(node):
    point = await node.remember(
        "Isolate the drive then purge the coolant line before re-homing the gantry",
        collection="procedural")
    scored = node.late_interaction.score("purge the coolant line", [point.id], explain=True)
    assert scored and scored[0].point_id == point.id
    alignments = scored[0].as_dict()["alignments"]
    assert alignments
    assert any(a["doc_term"] in {"purge", "coolant", "line"} for a in alignments)


@pytest.mark.asyncio
async def test_late_interaction_storage_is_quantized(node):
    point = await node.remember("a reasonably long procedural memory about the gantry drive")
    tokens = len(node.late_interaction.terms[point.id])
    # int8 codes + one float scale per token, not float32 per dimension
    assert node.late_interaction.resident_bytes < tokens * node.embedder.dim * 4


def test_common_words_are_not_corrected_into_domain_terms(understanding):
    """'last' must not become 'mast' just because the corpus mentions a mast."""
    for text in ["the west mast was inspected", "mast sensor replaced"] * 3:
        understanding.observe(text)
    analysis = understanding.analyse("readings from the last shift")
    assert "last" not in analysis.corrections
    assert "shift" not in analysis.corrections


def test_expansions_are_words_not_numbers(understanding):
    understanding.observe("fault reported at 12.97123,77.59456 on the west mast")
    analysis = understanding.analyse("fault on the mast")
    assert all(any(c.isalpha() for c in term) for term in analysis.expansions)
