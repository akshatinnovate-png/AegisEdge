"""Egress scheduling: what goes first when the link will not last."""
from __future__ import annotations

import pytest

from aegis.core.clock import HybridClock
from aegis.sync.crdt import Operation, OpKind, OpLog
from aegis.sync.egress import EgressPlanner


def _op(point_id: str, clock: HybridClock, kind: OpKind = OpKind.UPSERT,
        body: dict | None = None, ts: float | None = None) -> Operation:
    op = Operation(kind=kind, point_id=point_id, hlc=clock.now().pack(),
                   device_id="edge-a", body=body if body is not None else {"text": "x" * 120})
    if ts is not None:
        op.ts = ts
    return op


def test_a_tombstone_that_reuses_the_points_clock_never_deletes_anything():
    """The defect: a delete stamped with the HLC of the write it deletes.

    Resolution is `dominates()` — strictly happens-after — so a tombstone that
    *ties* with the upsert loses. The author's own log rejected its own delete,
    and the peer filed it as a concurrent write for a human to arbitrate. The
    memory stayed on every device, and the cycle reported CONVERGED.
    """
    clock = HybridClock("edge-a")
    author, peer = OpLog("edge-a"), OpLog("edge-b")
    upsert = _op("p1", clock)
    author.append(upsert)
    peer.merge([upsert])

    stale = Operation(kind=OpKind.DELETE, point_id="p1", hlc=upsert.hlc, device_id="edge-a")
    author.append(stale)
    accepted, conflicted, _ = peer.merge([stale])
    assert author.tombstones == {}          # the author rejected its own delete
    assert accepted == [] and len(conflicted) == 1
    assert peer.tombstones == {}            # and the peer still has the memory

    fresh = Operation(kind=OpKind.DELETE, point_id="p1", hlc=clock.now().pack(),
                      device_id="edge-a")
    author.append(fresh)
    accepted, conflicted, _ = peer.merge([fresh])
    assert len(author.tombstones) == 1 and len(peer.tombstones) == 1
    assert len(accepted) == 1 and conflicted == []


@pytest.mark.asyncio
async def test_a_delete_is_stamped_now_so_it_outranks_the_write_it_removes(node):
    from aegis.core.clock import HLC

    point = await node.remember("a secret that must be forgotten", "episodic")
    written = HLC.parse(point.hlc)
    op = node.sync.record_local(point, OpKind.DELETE)
    assert op is not None
    assert op.clock.dominates(written), "the tombstone does not outrank the write"
    assert len(node.sync.oplog.tombstones) == 1, "the node rejected its own tombstone"
    assert node.sync.oplog.rejected_stale == 0


def test_the_merge_is_order_independent_which_is_what_lets_egress_reorder():
    """The scheduler reorders egress. It may only do that because this holds."""
    clock = HybridClock("edge-a")
    ops = [_op("p1", clock), _op("p2", clock), _op("p1", clock), _op("p3", clock)]
    ops.append(Operation(kind=OpKind.DELETE, point_id="p2", hlc=clock.now().pack(),
                         device_id="edge-a"))

    def final_state(sequence: list[Operation]) -> dict[str, str]:
        log = OpLog("edge-b")
        for op in sequence:
            log.merge([op])
        return {pid: clk.pack() for pid, clk in log.heads.items()}, set(log.tombstones)

    forward = final_state(ops)
    backward = final_state(list(reversed(ops)))
    shuffled = final_state([ops[i] for i in (2, 0, 4, 3, 1)])
    assert forward == backward == shuffled


def test_an_older_write_for_the_same_point_is_not_sent_at_all():
    clock = HybridClock("edge-a")
    first, second = _op("p1", clock), _op("p1", clock)
    plan = EgressPlanner("edge-a").plan([first, second], link="healthy")
    assert [op.op_id for op in plan.send] == [second.op_id]
    assert [op.op_id for op in plan.redundant] == [first.op_id]
    # and it is released only against the acknowledgement of the one that replaced it
    assert plan.releases == {second.op_id: [first.op_id]}
    assert plan.redundant_bytes > 0


def test_a_tombstone_is_never_suppressed_as_redundant():
    """LWW makes an older upsert a no-op. It does not do that to a delete."""
    clock = HybridClock("edge-a")
    delete = _op("p1", clock, OpKind.DELETE, body={})
    later = _op("p1", clock)                     # a re-ingest after the delete
    plan = EgressPlanner("edge-a").plan([delete, later], link="healthy")
    sent = {op.op_id for op in plan.send}
    assert delete.op_id in sent, "a delete was dropped before it ever left the device"
    assert plan.redundant == []


def test_obligations_go_before_conveniences():
    clock = HybridClock("edge-a")
    chatter = [_op(f"p{i}", clock) for i in range(6)]
    delete = _op("px", clock, OpKind.DELETE, body={})
    plan = EgressPlanner("edge-a").plan(chatter + [delete], link="healthy")
    assert plan.send[0].op_id == delete.op_id
    assert "wrong state" in plan.assessments[0].reasons[0]


def test_a_budget_is_respected_but_never_sends_nothing():
    clock = HybridClock("edge-a")
    ops = [_op(f"p{i}", clock) for i in range(20)]
    planner = EgressPlanner("edge-a")
    plan = planner.plan(ops, budget_bytes=600, link="metered")
    assert plan.planned_bytes <= 600 or len(plan.send) == 1
    assert plan.send, "a budget smaller than one operation must still make progress"
    assert len(plan.deferred) == len(ops) - len(plan.send)

    # an offline link is the one case where nothing goes
    offline = planner.plan(ops, link="offline")
    assert offline.send == [] and len(offline.deferred) == len(ops)


def test_nothing_waits_longer_than_the_starvation_ceiling():
    """Value ordering alone would let a dull operation wait forever."""
    clock = HybridClock("edge-a")
    planner = EgressPlanner("edge-a")
    dull = _op("old", clock, body={"text": "y" * 400}, ts=1_000.0)
    urgent = [_op(f"p{i}", clock, OpKind.DELETE, body={}) for i in range(4)]
    now = 1_000.0 + planner.STARVATION_S + 1
    plan = planner.plan([dull] + urgent, link="healthy", now=now)
    assert plan.send[0].op_id == dull.op_id
    assert plan.assessments[0].starving
    # and before the ceiling it stays behind the obligations
    early = planner.plan([dull] + urgent, link="healthy", now=1_000.0 + 1)
    assert early.send[0].op_id != dull.op_id


def test_fifo_is_available_as_the_control_and_orders_differently():
    clock = HybridClock("edge-a")
    planner = EgressPlanner("edge-a")
    ops = [_op(f"p{i}", clock) for i in range(4)]
    ops.append(_op("px", clock, OpKind.DELETE, body={}))
    value_first = planner.plan(ops, link="healthy", order="value").send
    fifo = planner.plan(ops, link="healthy", order="fifo").send
    assert value_first[0].kind is OpKind.DELETE
    assert fifo[0].kind is OpKind.UPSERT
    assert {op.op_id for op in value_first} == {op.op_id for op in fifo}   # same set


@pytest.mark.asyncio
async def test_what_the_budget_defers_stays_in_the_durable_queue(node):
    """Deferring is not dropping. The queue is the record, and it must hold.

    The node is deliberately not started: its own sync loop would drain the
    queue in the background and this test would pass for the wrong reason.
    """
    for i in range(12):
        await node.remember(f"observation {i} about the line", "episodic")
    depth = node.sync.queue.depth
    assert depth > 0
    # every live state, because which one the oracle reports depends on
    # whether its probe has run yet and this test is not about that
    node.sync.egress.BUDGETS = {**node.sync.egress.BUDGETS,
                                "healthy": 900, "metered": 900, "degraded": 900}
    await node.sync.reconcile()
    assert node.sync.queue.depth > 0, "a budgeted cycle drained the whole queue"
    assert node.sync.queue.depth < depth, "a budgeted cycle sent nothing"
    # and the rest goes on the next cycles, without a byte lost
    for _ in range(12):
        if not node.sync.queue.depth:
            break
        await node.sync.reconcile()
    assert node.sync.queue.depth == 0
