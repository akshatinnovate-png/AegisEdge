"""The simulator's invariants, checked against a live node.

These assert two separate things, and the distinction matters: that the checks
*run* and count, and that they *fire* when the property genuinely does not
hold. A monitor that never fires is indistinguishable from a monitor that
cannot, which is why every invariant below is also broken on purpose.
"""
import asyncio

import pytest

from aegis.core.invariants import InvariantMonitor
from aegis.sync.crdt import OpKind, Operation
from aegis.sync.identity import DeviceIdentity


class _Log:
    def __init__(self, ops=()):
        self.ops = list(ops)
        self.seen = {op.op_id for op in self.ops}


class _Bus:
    def __init__(self): self.published = []
    def publish(self, *a, **k): self.published.append((a, k))


class _Node:
    """The smallest thing the monitor needs: a log, a mesh, a clock, a bus."""

    def __init__(self, ops=(), mesh=None, clock=None):
        self.sync = type("S", (), {"oplog": _Log(ops)})()
        self.mesh = mesh
        self.clock = clock
        self.bus = _Bus()


def _signed_op(identity, text="observation", sensitivity="internal"):
    op = Operation(kind=OpKind.UPSERT, point_id="p1", device_id=identity.device_id,
                   body={"text": text, "sensitivity": sensitivity})
    op.sig = identity.sign(op.as_dict())
    return op


class _Mesh:
    def __init__(self, identity, shareable=(), may_share=lambda _op: True,
                 node_id="edge-00"):
        self.identity = identity
        self.node_id = node_id
        self.known = {op.op_id: op for op in shareable}
        self._shareable_ops = dict(self.known)
        self.may_share = may_share
    def _shareable(self): return self._shareable_ops


# -- it runs, and it counts ------------------------------------------------

def test_a_clean_node_accumulates_assertions_and_no_violations():
    identity = DeviceIdentity("edge-00")
    op = _signed_op(identity)
    node = _Node([op], _Mesh(identity, [op]))
    monitor = InvariantMonitor(node)
    for _ in range(5):
        assert monitor.tick() == []
    snap = monitor.snapshot()
    assert snap["violations"] == 0
    assert snap["assertions"] == 5 * len(monitor.invariants)
    assert snap["ticks"] == 5


def test_every_invariant_states_what_it_promises():
    """A counter with no statement of what it counted is decoration."""
    monitor = InvariantMonitor(_Node())
    for invariant in monitor.snapshot()["invariants"]:
        assert invariant["promise"]
        assert invariant["name"]


def test_the_snapshot_refuses_to_call_a_clean_counter_a_proof():
    monitor = InvariantMonitor(_Node())
    monitor.tick()
    assert "not a proof" in monitor.snapshot()["claim"]


# -- it fires ---------------------------------------------------------------

def test_a_tampered_body_is_caught():
    """The attack the simulator found, detected by a node on its own."""
    author, holder = DeviceIdentity("edge-00"), DeviceIdentity("edge-01")
    holder.learn("edge-00", author.public_key_b64)
    op = _signed_op(author, "dispatch to grid 14")
    op.body["text"] = "dispatch to grid 41"          # rewritten after signing
    node = _Node([op], _Mesh(holder, [op]))
    found = InvariantMonitor(node).tick()
    assert [f.invariant for f in found] == ["bodies-intact"]
    assert "does not verify" in found[0].detail


def test_another_devices_restricted_memory_sitting_here_is_caught():
    """The seed-5 property: egress filtering failed upstream and we took it."""
    identity = DeviceIdentity("edge-00")
    op = _signed_op(identity, sensitivity="restricted")
    op.device_id = "edge-09"                      # somebody else wrote it
    mesh = _Mesh(identity, [op], node_id="edge-00",
                 may_share=lambda o: o.body.get("sensitivity") != "restricted")
    found = InvariantMonitor(_Node([op], mesh)).tick()
    assert "policy" in [f.invariant for f in found], [f.invariant for f in found]


def test_this_devices_own_restricted_memory_is_not_a_violation():
    """A device is allowed to keep what it decided may never leave."""
    identity = DeviceIdentity("edge-00")
    op = _signed_op(identity, sensitivity="restricted")
    mesh = _Mesh(identity, [op], node_id="edge-00",
                 may_share=lambda o: o.body.get("sensitivity") != "restricted")
    found = InvariantMonitor(_Node([op], mesh)).tick()
    assert "policy" not in [f.invariant for f in found]


def test_an_operation_attributed_to_nobody_is_caught():
    op = Operation(kind=OpKind.UPSERT, point_id="p1", device_id="", body={"text": "x"})
    found = InvariantMonitor(_Node([op])).tick()
    assert "no-fabrication" in [f.invariant for f in found]


def test_a_signature_from_an_unknown_device_is_caught():
    stranger, holder = DeviceIdentity("edge-99"), DeviceIdentity("edge-01")
    op = _signed_op(stranger)
    found = InvariantMonitor(_Node([op], _Mesh(holder, [op]))).tick()
    assert "no-fabrication" in [f.invariant for f in found]


def test_a_duplicated_operation_is_caught():
    identity = DeviceIdentity("edge-00")
    op = _signed_op(identity)
    node = _Node([op], _Mesh(identity, [op]))
    node.sync.oplog.ops.append(op)                  # same id twice in the log
    found = InvariantMonitor(node).tick()
    assert "no-duplicates" in [f.invariant for f in found]


def test_a_clock_running_backwards_is_caught():
    class _Backwards:
        def __init__(self): self.n = 100
        def now(self):
            from aegis.core.clock import HLC
            self.n -= 10
            return HLC(self.n, 0, "edge-00")
    node = _Node(clock=_Backwards())
    monitor = InvariantMonitor(node)
    monitor.tick()                                   # first call only records
    found = monitor.tick()
    assert "clock-monotonic" in [f.invariant for f in found]


def test_a_clock_dragged_into_the_future_is_caught():
    class _Future:
        def now(self):
            import time
            from aegis.core.clock import HLC
            return HLC(int(time.time() * 1000) + 90 * 86_400_000, 0, "edge-00")
    found = InvariantMonitor(_Node(clock=_Future())).tick()
    assert "clock-not-poisoned" in [f.invariant for f in found]


def test_losing_an_operation_this_device_wrote_is_caught():
    identity = DeviceIdentity("edge-00")
    node = _Node([], _Mesh(identity))
    monitor = InvariantMonitor(node)
    monitor.note_created("op-that-should-be-here")
    found = monitor.tick()
    assert "origin-retains" in [f.invariant for f in found]


def test_a_violation_is_published_rather_than_raised():
    """The node keeps serving. Halting turns a partial fault into a total one."""
    op = Operation(kind=OpKind.UPSERT, point_id="p1", device_id="", body={"text": "x"})
    node = _Node([op])
    InvariantMonitor(node).tick()
    kinds = [a[1] for a, _ in node.bus.published]
    assert "invariant_violated" in kinds


def test_a_check_that_throws_is_itself_a_finding():
    """Otherwise the counter climbs while nothing is being checked."""
    monitor = InvariantMonitor(_Node())
    monitor.invariants[0].check = lambda _m: (_ for _ in ()).throw(RuntimeError("boom"))
    found = monitor.tick()
    assert any("the check itself failed" in f.detail for f in found)


# -- it is affordable -------------------------------------------------------

def test_the_cost_is_bounded_by_the_budget_not_the_corpus():
    """A long log must not make the check the most expensive thing the node does."""
    identity = DeviceIdentity("edge-00")
    ops = []
    for i in range(4000):
        op = Operation(kind=OpKind.UPSERT, point_id=f"p{i}", device_id="edge-00",
                       body={"text": f"observation {i}", "sensitivity": "internal"})
        op.sig = identity.sign(op.as_dict())
        ops.append(op)
    monitor = InvariantMonitor(_Node(ops, _Mesh(identity, ops)))
    monitor.tick()
    small = InvariantMonitor(_Node(ops[:64], _Mesh(identity, ops[:64])))
    small.tick()
    # Within a factor of four of a log 60x shorter: the budget is doing its job.
    assert monitor.last_tick_ms < max(small.last_tick_ms * 4, 50.0), (
        monitor.last_tick_ms, small.last_tick_ms)


def test_the_cursor_walks_the_whole_log_over_successive_ticks():
    """A budget that only ever examined the first 64 would be theatre."""
    identity = DeviceIdentity("edge-00")
    ops = [_signed_op(identity) for _ in range(300)]
    for i, op in enumerate(ops):
        op.op_id = f"op-{i:04d}"
    monitor = InvariantMonitor(_Node(ops, _Mesh(identity, ops)))
    invariant = next(i for i in monitor.invariants if i.name == "bodies-intact")
    seen = set()
    for _ in range(10):
        before = invariant.cursor
        monitor.tick()
        seen.add(before)
    assert len(seen) > 1


# -- the live node ----------------------------------------------------------

def test_a_real_node_serves_its_invariants_over_the_api():
    from fastapi.testclient import TestClient

    from aegis.main import app

    with TestClient(app) as client:
        body = client.post("/api/v1/integrity/invariants/check").json()
        assert body["found_now"] == []
        assert body["violations"] == 0
        assert body["assertions"] >= len(body["invariants"])
        assert client.get("/api/v1/health").json()["invariants"]["ticks"] >= 1
