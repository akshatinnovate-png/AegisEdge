"""Sync: offline queueing, convergence, resumption, conflicts, policy egress."""
from __future__ import annotations

import pytest

from aegis.sync.crdt import OpKind, OpLog, Operation
from aegis.sync.engine import SyncState
from aegis.sync.merkle import MerkleTree
from aegis.sync.oracle import LinkState


def test_merkle_roots_match_until_one_side_diverges():
    a, b = MerkleTree(8), MerkleTree(8)
    items = [(f"p{i}", f"v{i}") for i in range(500)]
    a.rebuild(items)
    b.rebuild(items)
    assert a.divergent(b.digest()) == []
    b.set("p123", "v123-newer")
    divergent = a.divergent(b.digest())
    assert len(divergent) == 1                      # one bucket, not 500 points
    assert a.bytes_saved(divergent) > 0


def test_oplog_replay_is_idempotent_and_lww():
    log = OpLog("A")
    op = Operation(kind=OpKind.UPSERT, point_id="p1", hlc="0000000001000.00000.A", device_id="A")
    assert log.append(op)
    assert not log.append(op)                        # replay applies once
    newer = Operation(kind=OpKind.UPSERT, point_id="p1", hlc="0000000002000.00000.B", device_id="B")
    accepted, conflicted, dropped = log.merge([newer])
    assert len(accepted) == 1 and not conflicted and not dropped
    older = Operation(kind=OpKind.UPSERT, point_id="p1", hlc="0000000000500.00000.C", device_id="C")
    accepted, conflicted, dropped = log.merge([older])
    assert not accepted and len(dropped) == 1        # a stale write never wins


def test_concurrent_writes_reach_the_arbiter():
    log = OpLog("A")
    log.append(Operation(kind=OpKind.UPSERT, point_id="p1", hlc="0000000002000.00000.A", device_id="A"))
    concurrent = Operation(kind=OpKind.UPSERT, point_id="p1", hlc="0000000001900.00000.B", device_id="B")
    _, conflicted, _ = log.merge([concurrent])
    assert len(conflicted) == 1


@pytest.mark.asyncio
async def test_restricted_memory_is_never_queued_for_egress(node):
    point = await node.remember("api_key=sk-live-3391 for the coordinator")
    assert point.sensitivity.value == "restricted"
    assert node.sync.record_local(point) is None
    assert all(op.point_id != point.id for op in node.sync.queue.pending)


@pytest.mark.asyncio
async def test_offline_work_queues_and_converges_on_reconnect(node):
    node.oracle.forced_offline = True
    await node.oracle.probe_once()
    assert node.oracle.state is LinkState.OFFLINE

    for i in range(5):
        await node.remember(f"observation {i} recorded while the link was down")
    assert node.sync.queue.depth == 5                # durable, nothing lost

    node.oracle.forced_offline = False
    await node.oracle.probe_once()                   # fires the restore callback
    result = await node.sync.reconcile(trigger="test")
    assert node.sync.state is SyncState.CONVERGED
    assert result["pushed"] >= 5
    assert node.sync.queue.depth == 0
    assert node.transport.received >= 5


@pytest.mark.asyncio
async def test_second_handshake_resumes_the_session(node):
    await node.sync.reconcile()
    first = node.sync.session
    await node.sync.reconcile()
    assert node.sync.session == first                # continuation, not a new session
    assert node.sync.zero_rtt
    assert node.sync.resumptions >= 1


@pytest.mark.asyncio
async def test_fleet_knowledge_is_pulled_down(node):
    await node.sync.reconcile()
    vector = node.embedder.embed_sync(["Torque limit on line 2 was raised to 42 Nm"])[0].tolist()
    node.transport.inject("fleet-pt-1", node.clock.now().pack(), {
        "collection": "semantic", "text": "Torque limit on line 2 was raised to 42 Nm",
        "dense": vector, "sensitivity": "internal", "confidence": 0.9, "device_id": "edge-99",
    })
    result = await node.sync.reconcile()
    assert result["pulled"] >= 1
    assert "fleet-pt-1" in node.store.points
    assert node.store.points["fleet-pt-1"].source == "fleet"


@pytest.mark.asyncio
async def test_interrupted_sync_resumes_from_cursor(node):
    for i in range(3):
        await node.remember(f"pre-partition observation {i}")
    node.transport.partition(True)
    result = await node.sync.reconcile()
    assert node.sync.state is SyncState.BACKOFF
    assert "error" in result
    assert node.sync.queue.depth == 3                # the lease was returned, not dropped

    node.transport.partition(False)
    result = await node.sync.reconcile()
    assert node.sync.state is SyncState.CONVERGED
    assert node.sync.queue.depth == 0


@pytest.mark.asyncio
async def test_link_recovers_within_a_single_probe(node):
    node.oracle.forced_offline = True
    await node.oracle.probe_once()
    assert node.oracle.state is LinkState.OFFLINE

    node.oracle.forced_offline = False
    await node.oracle.probe_once()
    assert node.oracle.state is not LinkState.OFFLINE      # not several probes later
    assert node.oracle.reconnect_ms is not None
    await node.oracle.probe_once()
    assert node.oracle.state is LinkState.HEALTHY


@pytest.mark.asyncio
async def test_a_single_dropped_probe_does_not_declare_an_outage(node):
    for _ in range(3):
        await node.oracle.probe_once()
    assert node.oracle.state is LinkState.HEALTHY
    node.transport.partition(True)
    await node.oracle.probe_once()
    node.transport.partition(False)
    assert node.oracle.state is not LinkState.OFFLINE      # slow to condemn the link
