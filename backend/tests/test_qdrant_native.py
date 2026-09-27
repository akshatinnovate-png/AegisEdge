"""The Qdrant-native hybrid path: one call, and a refusal that stays a refusal."""
from __future__ import annotations

import numpy as np
import pytest

from aegis.memory.filters import Condition, Filter, Op
from aegis.memory.qdrant_native import LATE, prune_late, translate


def test_a_filter_with_no_faithful_qdrant_form_is_refused_not_widened():
    # CONTAINS reads substrings and list membership in our payloads; Qdrant has
    # no single condition that means both. Translating it loosely would answer
    # a filtered query with unfiltered results, so it must return None.
    assert translate(Filter(must=[Condition("text", Op.CONTAINS, "pump")])) is None
    assert translate(Filter(must=[Condition("source", Op.EXISTS, True)])) is None
    assert translate(Filter(should=[Condition("device_id", Op.NE, "edge-01")])) is None


def test_translation_keeps_the_meaning_of_every_op_it_accepts():
    spec = Filter(must=[Condition("sensitivity", Op.EQ, "secret"),
                        Condition("ts", Op.GTE, 10.0),
                        Condition("device_id", Op.NE, "edge-09")],
                  should=[Condition("collection", Op.IN, ["episodic", "semantic"])],
                  must_not=[Condition("stale", Op.EQ, True)])
    translated = translate(spec)
    assert translated is not None
    # `!=` is not a Qdrant condition — it is a negated match, and it has to
    # cross the tree to stay true rather than be dropped on the floor.
    assert len(translated.must) == 2
    assert {c.key for c in translated.must_not} == {"device_id", "stale"}
    assert len(translated.should) == 1


def test_an_empty_filter_translates_to_no_filter_not_to_a_refusal():
    assert translate(None) is None
    assert translate(Filter()) is None


def test_the_late_budget_keeps_the_least_generic_tokens():
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(24, 16)).astype(np.float32)
    # one token deliberately made into the document's own centroid: it is the
    # token any passage would have supplied, so it is the one to drop first
    vectors[5] = vectors.mean(axis=0) * 4
    pruned = prune_late(vectors, 8)
    assert pruned.shape == (8, 16)
    assert not any(np.allclose(row, vectors[5]) for row in pruned)
    # reading order survives pruning
    order = [int(np.argmin(np.linalg.norm(vectors - row, axis=1))) for row in pruned]
    assert order == sorted(order)


def test_the_late_budget_is_a_no_op_below_the_budget():
    rng = np.random.default_rng(1)
    vectors = rng.normal(size=(3, 8)).astype(np.float32)
    assert np.array_equal(prune_late(vectors, 8), vectors)
    assert prune_late(np.zeros((0, 8), dtype=np.float32), 4).shape == (0, 8)


@pytest.fixture
def store(tmp_path):
    from aegis.memory.vectorstore import QdrantStore

    opened = QdrantStore(16, ("episodic",), str(tmp_path), late_tokens=4)
    yield opened
    opened.close()


def _fill(store, count: int = 30):
    from aegis.memory.schema import MemoryPoint

    rng = np.random.default_rng(3)
    store.late_provider = lambda text: rng.normal(size=(12, 16)).astype(np.float32)
    for i in range(count):
        store.upsert(MemoryPoint(id=f"m{i}", collection="episodic",
                                 text=f"pump {i} coolant pressure",
                                 dense=rng.normal(size=16).astype(np.float32),
                                 sparse={i: 1.0, 7: 0.5, 500 + i: 0.25}))
    return rng


def test_the_whole_hybrid_pipeline_is_one_engine_call(store):
    rng = _fill(store)
    hits, plan = store.search_native("episodic", rng.normal(size=16).astype(np.float32),
                                     {3: 1.0, 7: 0.4}, 5)
    assert hits and not plan.fell_back
    assert plan.calls == 1                       # four stages, one round trip
    assert plan.fusion == "rrf"
    assert [s.name for s in plan.stages] == ["dense recall", "sparse recall", "fusion"]
    assert all(s.where == "engine" for s in plan.stages)


def test_a_refused_query_returns_a_reason_and_never_a_silent_empty(store):
    _fill(store)
    hits, plan = store.search_native("episodic", np.zeros(16, dtype=np.float32), {1: 1.0}, 5,
                                     spec=Filter(must=[Condition("text", Op.CONTAINS, "pump")]))
    assert hits == []
    assert plan.fell_back                        # the caller reads this, not the empty list
    assert store.native_refusals == 1


def test_a_collection_on_the_old_single_vector_schema_keeps_the_python_path(tmp_path):
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    from aegis.memory.vectorstore import QdrantStore

    legacy = QdrantClient(path=str(tmp_path / "qdrant"))
    legacy.create_collection("episodic",
                             vectors_config=VectorParams(size=16, distance=Distance.COSINE))
    legacy.close()

    store = QdrantStore(16, ("episodic",), str(tmp_path))
    try:
        assert store.hybrid.schema["episodic"] == "legacy"
        assert not store.hybrid.native("episodic")
        _, plan = store.search_native("episodic", np.zeros(16, dtype=np.float32), {1: 1.0}, 3)
        assert "legacy" in plan.fell_back        # refused, and the data is untouched
    finally:
        store.close()


def test_late_interaction_reranks_inside_the_engine(store):
    rng = _fill(store)
    assert store.hybrid.has_late("episodic")
    hits, plan = store.search_native("episodic", rng.normal(size=16).astype(np.float32),
                                     {3: 1.0, 7: 0.4}, 5,
                                     late=rng.normal(size=(6, 16)).astype(np.float32))
    assert hits and not plan.fell_back
    assert plan.rerank.startswith("maxsim")
    assert plan.calls == 1
    assert LATE in store.hybrid.snapshot()["vectors"]


def test_the_bake_off_reports_a_disagreement_rather_than_hiding_one(store):
    rng = _fill(store)
    report = store.bake_off("episodic", rng.normal(size=16).astype(np.float32), {3: 1.0, 7: 0.4}, 5)
    assert report["k"] == 5 and report["backend"] == "qdrant-local"
    assert 0.0 <= report["overlap"] <= 1.0
    assert 0.0 <= report["rank_agreement"] <= 1.0
    assert report["local_ms"] > 0 and report["native_ms"] > 0
    assert report["plan"]["calls"] == 1


def _legacy_store(tmp_path, dim: int = 16):
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    from aegis.memory.vectorstore import QdrantStore

    client = QdrantClient(path=str(tmp_path / "qdrant"))
    client.create_collection("episodic",
                             vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
    client.upsert("episodic", points=[
        PointStruct(id=i, vector=[0.1] * dim, payload={"aegis_id": f"old{i}"}) for i in range(4)])
    client.close()
    return QdrantStore(dim, ("episodic",), str(tmp_path))


def test_migration_moves_a_legacy_collection_onto_the_hybrid_schema(tmp_path):
    from aegis.memory.schema import MemoryPoint

    store = _legacy_store(tmp_path)
    try:
        rng = np.random.default_rng(11)
        points = [MemoryPoint(id=f"old{i}", collection="episodic", text=f"note {i}",
                              dense=rng.normal(size=16).astype(np.float32),
                              sparse={i: 1.0}) for i in range(4)]
        report = store.migrate_schema(points)
        assert report["schema"]["episodic"] == "hybrid"
        assert report["migrated"][0]["rewritten"] == 4
        # and the engine path is open afterwards, on the migrated data
        hits, plan = store.search_native("episodic", points[0].dense, {0: 1.0}, 3)
        assert hits and not plan.fell_back
    finally:
        store.close()


def test_migration_refuses_to_delete_points_only_qdrant_holds(tmp_path):
    from aegis.memory.schema import MemoryPoint

    store = _legacy_store(tmp_path)
    try:
        # Qdrant holds four; the node can only supply one. Recreating the
        # collection would destroy the other three, so this must refuse.
        offered = [MemoryPoint(id="old0", collection="episodic", text="note 0",
                               dense=np.zeros(16, dtype=np.float32), sparse={0: 1.0})]
        report = store.migrate_schema(offered)
        assert report["migrated"] == [] and report["points"] == 0
        assert report["refused"][0]["in_qdrant"] == 4
        assert report["refused"][0]["offered"] == 1
        assert store.hybrid.schema["episodic"] == "legacy"      # untouched

        forced = store.migrate_schema(offered, force=True)
        assert forced["migrated"][0]["forced"] is True
    finally:
        store.close()
