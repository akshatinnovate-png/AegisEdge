"""Memory: quantized recall, WAL recovery, tiering, consolidation."""
from __future__ import annotations

import numpy as np
import pytest

from aegis.memory.ann import CostModel, Strategy
from aegis.memory.index import CollectionIndex
from aegis.memory.schema import MemoryPoint, Tier
from aegis.memory.tiering import TieringPolicy
from aegis.memory.wal import WriteAheadLog


def _unit(seed: int, dim: int = 64) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _index(tmp_path, dim: int = 64) -> CollectionIndex:
    return CollectionIndex(dim, CostModel().calibrate(dim=dim, sample=512), "episodic", tmp_path)


def test_quantized_tiers_still_return_the_right_neighbour(tmp_path):
    index = _index(tmp_path)
    vectors = {}
    for i in range(90):
        v = _unit(i)
        vectors[f"p{i}"] = v
        index.upsert(f"p{i}", v, {i % 7: 1.0}, [Tier.HOT, Tier.WARM, Tier.COLD][i % 3],
                     {"collection": "episodic", "ts": float(i)})
    for probe in ("p11", "p44", "p77"):          # one point in each tier
        assert index.search_dense(vectors[probe], 3)[0][0] == probe


def test_cold_tier_vectors_leave_ram(tmp_path):
    index = _index(tmp_path)
    for i in range(40):
        index.upsert(f"p{i}", _unit(i), {}, Tier.HOT, {"collection": "episodic"})
    resident_before = index.storage.snapshot()["resident_bytes"]
    for i in range(30):
        index.move(f"p{i}", Tier.COLD)
    after = index.storage.snapshot()
    assert after["resident_bytes"] < resident_before        # actually evicted, not "compressed"
    assert after["cold_on_disk"] == 30
    assert index.search_dense(_unit(3), 1)[0][0] == "p3"    # still retrievable, paged in
    assert index.storage.page_ins > 0


def test_moving_a_point_between_tiers_preserves_recall(tmp_path):
    index = _index(tmp_path)
    for i in range(30):
        index.upsert(f"p{i}", _unit(i), {}, Tier.HOT, {"collection": "episodic"})
    assert index.move("p7", Tier.COLD)
    assert index.search_dense(_unit(7), 1)[0][0] == "p7"
    assert index.counts()["cold"] == 1
    assert index.move("p7", Tier.HOT)                       # promotion brings it back
    assert index.storage.row_of.get("p7") is not None


def test_removing_a_point_clears_every_structure(tmp_path):
    index = _index(tmp_path)
    for i in range(20):
        index.upsert(f"p{i}", _unit(i), {i: 1.0}, Tier.HOT, {"collection": "episodic"})
    index.remove("p5")
    assert len(index) == 19
    assert index.search_sparse({5: 1.0}, 3) == []
    assert "p5" not in index.storage.row_of
    assert all(pid != "p5" for pid, _ in index.search_dense(_unit(5), 5))


def test_wal_replays_and_truncates_a_torn_tail(tmp_path):
    wal = WriteAheadLog(tmp_path / "m.wal")
    for i in range(5):
        wal.append("upsert", {"id": f"p{i}"})
    with open(wal.path, "a", encoding="utf-8") as fh:
        fh.write("deadbeef {not json at all\n")     # simulate a crash mid-write
    reopened = WriteAheadLog(tmp_path / "m.wal")
    records = list(reopened.replay())
    assert len(records) == 5
    assert reopened.torn == 1


def test_tiering_pins_and_evicts():
    policy = TieringPolicy(hot_capacity=1, warm_capacity=1)
    hot = MemoryPoint(text="hot", access_count=100)
    pinned = MemoryPoint(text="pinned", pinned=True)
    cold = MemoryPoint(text="cold", access_count=0, confidence=0.1)
    expired = MemoryPoint(text="expired", ttl_s=0.0, created_at=0.0)
    plan = policy.plan([hot, pinned, cold, expired])
    assert plan[pinned.id] is Tier.HOT
    assert plan[expired.id] is Tier.EVICTED
    assert plan[cold.id] in {Tier.WARM, Tier.COLD}


@pytest.mark.asyncio
async def test_ingest_classifies_and_governs(node):
    point = await node.remember("Call operator 4471 at jo@plant.io about the alarm")
    assert point.sensitivity.value == "restricted"
    assert point.sync_class.value == "local_only"
    assert node.store.wal.stats()["appended"] > 0


@pytest.mark.asyncio
async def test_consolidation_distils_duplicates(node):
    for _ in range(4):
        await node.remember("Bay 3 conveyor vibration crossed 4.2 mm/s at 02:14")
    report = await node.consolidator.run("episodic")
    assert report.absorbed >= 3
    assert report.created >= 1
