"""Memory: quantized recall, WAL recovery, tiering, consolidation."""
from __future__ import annotations

import numpy as np
import pytest

from aegis.memory.index import TieredIndex
from aegis.memory.schema import MemoryPoint, Tier
from aegis.memory.tiering import TieringPolicy
from aegis.memory.wal import WriteAheadLog


def _unit(seed: int, dim: int = 64) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def test_quantized_tiers_still_return_the_right_neighbour():
    index = TieredIndex(64)
    vectors = {}
    for i in range(90):
        v = _unit(i)
        vectors[f"p{i}"] = v
        index.upsert(f"p{i}", v, {i % 7: 1.0}, [Tier.HOT, Tier.WARM, Tier.COLD][i % 3])
    for probe in ("p11", "p44", "p77"):          # one point in each tier
        assert index.search_dense(vectors[probe], 3)[0][0] == probe


def test_moving_a_point_between_tiers_preserves_recall():
    index = TieredIndex(64)
    for i in range(30):
        index.upsert(f"p{i}", _unit(i), {}, Tier.HOT)
    assert index.move("p7", Tier.COLD)
    assert index.search_dense(_unit(7), 1)[0][0] == "p7"
    assert index.counts()["cold"] == 1


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
