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
    # cross the tree to stay true rather than be dropped on the floor. It
    # crosses as *two* conditions: not-this-value, and not-absent, because
    # Qdrant's negated match alone would also return points without the field.
    assert len(translated.must) == 2
    keys = {c.key if hasattr(c, "key") else c.is_empty.key for c in translated.must_not}
    assert keys == {"device_id", "stale"}
    assert len(translated.must_not) == 3
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


def test_a_projection_says_how_much_it_threw_away():
    """A 2-D picture of a 256-D space is a compression. It has to admit it."""
    from aegis.retrieval.projection import project, scale

    rng = np.random.default_rng(5)
    vectors = rng.normal(size=(200, 64)).astype(np.float32)
    projection = project(vectors)
    assert projection.coords.shape == (200, 2)
    # Isotropic noise in 64 dimensions cannot have most of its variance in two.
    assert 0.0 < projection.explained < 0.2

    query = projection.project(rng.normal(size=64).astype(np.float32))
    placed = scale(projection.coords, query)
    assert len(placed["points"]) == 200
    # everything, the query included, lands inside the unit box it defines
    assert all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in placed["points"])
    assert all(0.0 <= v <= 1.0 for v in placed["query"])


def test_the_query_is_projected_through_the_corpus_basis_not_refitted():
    """Refitting to include the query would bend the space around it."""
    from aegis.retrieval.projection import project

    rng = np.random.default_rng(6)
    vectors = rng.normal(size=(60, 16)).astype(np.float32)
    projection = project(vectors)
    first = projection.coords.copy()
    projection.project(rng.normal(size=16).astype(np.float32) * 50)   # a far-away query
    assert np.array_equal(projection.coords, first)


def test_a_projection_of_nothing_is_not_a_projection_of_something():
    from aegis.retrieval.projection import project, scale

    empty = project(np.zeros((0, 8), dtype=np.float32))
    assert empty.coords.shape == (0, 2) and empty.explained == 0.0
    assert scale(empty.coords)["points"] == []
    # One point has no variance to explain, and must not claim it has all of it.
    assert project(np.ones((1, 8), dtype=np.float32)).explained == 0.0


def test_the_map_is_sampled_from_what_qdrant_holds(store):
    rng = _fill(store, 25)
    ids, vectors, payloads = store.hybrid.sample("episodic", 50)
    assert len(ids) == 25 and vectors.shape == (25, 16) and len(payloads) == 25
    assert all(pid.startswith("m") for pid in ids)
    assert payloads[0]["text"].startswith("pump")
    assert rng is not None


def test_turning_on_late_tokens_does_not_break_writes_to_older_collections(tmp_path):
    """The setting changed; the collection did not. Writes are not where to find out.

    Attaching the late vector because `late_tokens` is set, rather than because
    the collection has one, made every upsert fail with
    `Not existing vector name error: late` the moment the setting was raised on
    an existing node.
    """
    from aegis.memory.schema import MemoryPoint
    from aegis.memory.vectorstore import QdrantStore

    store = QdrantStore(16, ("episodic",), str(tmp_path))
    store.close()

    reopened = QdrantStore(16, ("episodic",), str(tmp_path), late_tokens=8)
    try:
        assert reopened.hybrid.wants_late("episodic")     # configured, not provisioned
        assert not reopened.hybrid.has_late("episodic")
        reopened.late_provider = lambda text: np.ones((12, 16), dtype=np.float32)
        point = MemoryPoint(id="m1", collection="episodic", text="pump",
                            dense=np.zeros(16, dtype=np.float32), sparse={1: 1.0})
        reopened.upsert(point)                            # must not raise

        report = reopened.migrate_schema([point])
        assert report["schema"]["episodic"] == "hybrid-late"
        assert "late-interaction vector" in report["reasons"]["episodic"]
        assert reopened.hybrid.has_late("episodic")
    finally:
        reopened.close()


def test_a_not_equals_filter_does_not_quietly_match_a_missing_field():
    """Qdrant's negated match also returns points without the field. Ours does not.

    Translating `x != v` to `must_not[match(x == v)]` alone widens the filter,
    which is the one thing this module promises never to do.
    """
    from qdrant_client import QdrantClient
    from qdrant_client import models as qm

    client = QdrantClient(location=":memory:")
    client.create_collection(
        "t", vectors_config={"dense": qm.VectorParams(size=4, distance=qm.Distance.COSINE)})
    rows = {"has-other": {"device_id": "edge-01"}, "has-match": {"device_id": "edge-09"},
            "absent": {}}
    client.upsert("t", points=[
        qm.PointStruct(id=i, vector={"dense": [1, 0, 0, 0]}, payload={"aegis_id": name, **pl})
        for i, (name, pl) in enumerate(rows.items())])

    spec = Filter(must=[Condition("device_id", Op.NE, "edge-09")])
    hits = client.query_points("t", query=[1, 0, 0, 0], using="dense", limit=10,
                               query_filter=translate(spec), with_payload=True).points
    engine = sorted(h.payload["aegis_id"] for h in hits)
    ours = sorted(name for name, payload in rows.items() if spec.matches(payload))
    assert engine == ours == ["has-other"]
