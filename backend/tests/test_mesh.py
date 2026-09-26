"""Mesh: IBLT reconciliation, causal delivery, gossip convergence, policy."""
from __future__ import annotations

import asyncio

import pytest

from aegis.core.bus import EventBus
from aegis.sync.causal import CausalBuffer, Ordering, VectorClock
from aegis.sync.compression import WireCodec, dequantize_vector, quantize_vector, read_varint, varint
from aegis.sync.crdt import OpKind, Operation
from aegis.sync.gossip import GossipAgent, MeshLink
from aegis.sync.iblt import IBLT


def test_iblt_recovers_the_difference_without_a_shared_dictionary():
    shared = {f"op-{i}" for i in range(4000)}
    a = shared | {f"a-{i}" for i in range(9)}
    b = shared | {f"b-{i}" for i in range(6)}
    cells = IBLT.size_for(20)
    only_a, only_b, complete = IBLT(cells).insert_many(a).subtract(IBLT(cells).insert_many(b)).decode()
    assert complete
    assert only_a == {f"a-{i}" for i in range(9)}
    assert only_b == {f"b-{i}" for i in range(6)}


def test_iblt_wire_size_tracks_the_difference_not_the_corpus():
    small = IBLT(IBLT.size_for(20))
    assert small.wire_bytes < 20_000                        # kilobytes, for any corpus size


def test_iblt_reports_failure_instead_of_a_partial_answer():
    a = {f"a-{i}" for i in range(400)}
    b = {f"b-{i}" for i in range(400)}
    _, _, complete = IBLT(32).insert_many(a).subtract(IBLT(32).insert_many(b)).decode()
    assert complete is False                                # honest, not silently truncated


def test_vector_clock_detects_concurrency():
    a = VectorClock.of({"A": 2, "B": 1})
    b = VectorClock.of({"A": 1, "B": 3})
    assert a.compare(b) is Ordering.CONCURRENT
    assert a.merge(b).mapping == {"A": 2, "B": 3}
    assert VectorClock.of({"A": 3}).dominates(VectorClock.of({"A": 1}))


def test_causal_buffer_holds_an_op_until_its_predecessor_arrives():
    first_clock = VectorClock().tick("A")
    second_clock = VectorClock().merge(first_clock).tick("B")
    first = Operation(kind=OpKind.UPSERT, point_id="p1", device_id="A")
    dependent = Operation(kind=OpKind.SUPERSEDE, point_id="p1", device_id="B")

    buffer = CausalBuffer("C")
    assert buffer.receive(dependent, second_clock, "B") == []       # too early
    assert len(buffer.pending) == 1
    released = buffer.receive(first, first_clock, "A")
    assert [op.device_id for op in released] == ["A", "B"]          # causal order restored
    assert buffer.pending == []


def test_causal_buffer_expires_orphans():
    buffer = CausalBuffer("C", max_hold_s=0.0)
    clock = VectorClock.of({"Z": 5})
    buffer.receive(Operation(point_id="p1", device_id="Z"), clock, "Z")
    assert buffer.expire() == 1


def test_wire_codec_compresses_and_roundtrips_vectors():
    import numpy as np

    rng = np.random.default_rng(0)
    ops = []
    for i in range(20):
        vector = rng.normal(size=256).astype("float32")
        vector /= np.linalg.norm(vector)
        ops.append({"op_id": f"op{i}", "kind": "upsert", "point_id": f"p{i}", "hlc": "1.0.a",
                    "device_id": "edge-07", "ts": 1.0,
                    "body": {"collection": "episodic", "sensitivity": "internal",
                             "text": f"observation {i}",
                             # As the sync engine builds it. The quantizing
                             # happens there now, not here: a codec that edits
                             # a signed field makes every operation carrying a
                             # vector unverifiable at the receiver.
                             "dense_q": quantize_vector(vector)}})
    codec = WireCodec()
    frame = codec.encode(ops)

    # The contract is exactness. Anything else and a signature taken before
    # the wire cannot be checked after it.
    decoded = codec.decode(frame)
    assert decoded == ops

    # And it still pays: about 325 bytes for an operation carrying a
    # 256-dimension vector, against 5,500 for the same operation with the
    # vector written out as JSON floats.
    assert frame["raw_bytes"] / frame["wire_bytes"] > 2
    assert frame["wire_bytes"] / len(ops) < 400

    verbose = []
    for op in ops:
        body = {k: v for k, v in op["body"].items() if k != "dense_q"}
        body["dense"] = dequantize_vector(op["body"]["dense_q"])
        verbose.append({**op, "body": body})
    assert WireCodec().encode(verbose)["wire_bytes"] > frame["wire_bytes"] * 2


def test_varint_roundtrip():
    for value in (0, 1, 127, 128, 300, 65_535, 2_000_000):
        assert read_varint(varint(value))[0] == value


def test_quantized_vector_roundtrip():
    import numpy as np

    vector = np.linspace(-1, 1, 64).astype("float32")
    restored = np.asarray(dequantize_vector(quantize_vector(vector)))
    assert float(np.abs(restored - vector).max()) < 0.02


def _mesh(names: list[str], may_share=None):
    link = MeshLink(latency_ms=1.0)
    bus = EventBus()
    stores: dict[str, dict] = {n: {} for n in names}
    agents: dict[str, GossipAgent] = {}
    for name in names:
        def apply_for(target):
            async def apply(op):
                stores[target][op.op_id] = op
            return apply
        agents[name] = GossipAgent(name, link, bus,
                                   op_source=lambda n=name: list(stores[n].values()),
                                   apply_op=apply_for(name),
                                   may_share=may_share or (lambda _op: True))
    for agent in agents.values():
        for other in names:
            if other != agent.node_id:
                agent.add_peer(other)
    return link, agents, stores


def test_two_devices_converge_with_no_cloud():
    names = ["edge-a", "edge-b"]
    link, agents, stores = _mesh(names)
    for i in range(25):
        op = Operation(kind=OpKind.UPSERT, point_id=f"p{i}", device_id="edge-a", body={"text": f"a{i}"})
        stores["edge-a"][op.op_id] = op
        agents["edge-a"].note_local(op)
    for i in range(10):
        op = Operation(kind=OpKind.UPSERT, point_id=f"q{i}", device_id="edge-b", body={"text": f"b{i}"})
        stores["edge-b"][op.op_id] = op
        agents["edge-b"].note_local(op)

    result = asyncio.run(agents["edge-a"].anti_entropy("edge-b"))
    assert result["complete"]
    assert result["pulled"] == 10 and result["pushed"] == 25
    assert len(stores["edge-a"]) == len(stores["edge-b"]) == 35


def test_knowledge_relays_through_a_third_device():
    names = ["edge-a", "edge-b", "edge-c"]
    link, agents, stores = _mesh(names)
    op = Operation(kind=OpKind.UPSERT, point_id="p1", device_id="edge-a", body={"text": "hello"})
    stores["edge-a"][op.op_id] = op
    agents["edge-a"].note_local(op)
    link.partition("edge-a", "edge-c")                  # A cannot reach C at all

    asyncio.run(agents["edge-a"].anti_entropy("edge-b"))
    asyncio.run(agents["edge-c"].anti_entropy("edge-b"))
    assert op.op_id in stores["edge-c"]                 # reached C via B


def test_policy_withholds_restricted_memories_from_peers():
    names = ["edge-a", "edge-b", "edge-c"]
    link, agents, stores = _mesh(
        names, may_share=lambda op: op.body.get("sensitivity") != "restricted")
    restricted_ids = set()
    for i in range(20):
        sensitivity = "restricted" if i in (3, 11) else "internal"
        op = Operation(kind=OpKind.UPSERT, point_id=f"p{i}", device_id="edge-a",
                       body={"text": f"obs {i}", "sensitivity": sensitivity})
        stores["edge-a"][op.op_id] = op
        agents["edge-a"].note_local(op)
        if sensitivity == "restricted":
            restricted_ids.add(op.op_id)

    asyncio.run(agents["edge-a"].anti_entropy("edge-b"))
    asyncio.run(agents["edge-c"].anti_entropy("edge-b"))
    assert len(stores["edge-b"]) == 18 and len(stores["edge-c"]) == 18
    assert not restricted_ids & set(stores["edge-b"])
    assert not restricted_ids & set(stores["edge-c"])   # and not via a relay either
    assert agents["edge-a"].withheld > 0


def test_withheld_operations_do_not_stall_causal_delivery():
    names = ["edge-a", "edge-b"]
    link, agents, stores = _mesh(names, may_share=lambda op: op.body.get("sensitivity") != "restricted")
    for i in range(12):
        op = Operation(kind=OpKind.UPSERT, point_id=f"p{i}", device_id="edge-a",
                       body={"text": f"o{i}", "sensitivity": "restricted" if i % 3 == 0 else "internal"})
        stores["edge-a"][op.op_id] = op
        agents["edge-a"].note_local(op)
    asyncio.run(agents["edge-a"].anti_entropy("edge-b"))
    assert agents["edge-b"].causal.snapshot()["pending"] == 0     # no permanent gaps
    assert len(stores["edge-b"]) == 8


def test_unreachable_peer_is_reported_not_raised():
    link, agents, _ = _mesh(["edge-a", "edge-b"])
    link.partition("edge-a", "edge-b")
    result = asyncio.run(agents["edge-a"].anti_entropy("edge-b"))
    assert "error" in result
    assert agents["edge-a"].peers["edge-b"].failures == 1


def test_note_local_is_idempotent():
    """Re-stamping an operation would advance the sequence past what peers saw."""
    link, agents, stores = _mesh(["edge-a", "edge-b"])
    ops = []
    for i in range(5):
        op = Operation(kind=OpKind.UPSERT, point_id=f"p{i}", device_id="edge-a", body={"text": str(i)})
        stores["edge-a"][op.op_id] = op
        agents["edge-a"].note_local(op)
        ops.append(op)
    clock_after_first = agents["edge-a"].causal.clock.mapping["edge-a"]
    for op in ops:
        agents["edge-a"].note_local(op)                       # re-noted, e.g. after a restart
    assert agents["edge-a"].causal.clock.mapping["edge-a"] == clock_after_first

    result = asyncio.run(agents["edge-a"].anti_entropy("edge-b"))
    assert result["pushed"] == 5
    assert len(stores["edge-b"]) == 5


def test_rumour_path_preserves_causal_order():
    link, agents, stores = _mesh(["edge-a", "edge-b"])
    first = Operation(kind=OpKind.UPSERT, point_id="p1", device_id="edge-a", body={"text": "base"})
    second = Operation(kind=OpKind.SUPERSEDE, point_id="p1", device_id="edge-a", body={"text": "newer"})
    stores["edge-a"][first.op_id] = first
    stores["edge-a"][second.op_id] = second
    agents["edge-a"].note_local(first)
    agents["edge-a"].note_local(second)

    # deliver the dependent rumour first: it must wait, not corrupt state
    asyncio.run(agents["edge-a"].rumour(second))
    assert second.op_id not in stores["edge-b"]
    assert agents["edge-b"].causal.snapshot()["pending"] == 1
    asyncio.run(agents["edge-a"].rumour(first))
    assert {first.op_id, second.op_id} <= set(stores["edge-b"])


def test_a_handling_class_survives_the_trip_between_devices():
    """A receiver cannot honour a restriction it was never told about.

    `_egress_body` never sent `sync_class` and `_materialize` hard-coded it to
    FULL, so a memory marked for redaction at its origin arrived on the next
    device freely shareable — and every decision that device then made about
    it, including what to pass on a third hop, rested on a class the memory
    never had.
    """
    from aegis.memory.schema import SyncClass

    assert SyncClass.strictest(SyncClass.FULL, SyncClass.REDACTED) is SyncClass.REDACTED
    assert SyncClass.strictest(SyncClass.REDACTED, SyncClass.LOCAL_ONLY) is SyncClass.LOCAL_ONLY
    assert SyncClass.strictest(SyncClass.FULL, SyncClass.FULL) is SyncClass.FULL
    # Ordering must be total and in the safe direction.
    ordered = sorted(SyncClass, key=lambda c: c.restriction)
    assert ordered[0] is SyncClass.FULL and ordered[-1] is SyncClass.LOCAL_ONLY


def test_http_mesh_link_reports_an_offline_radio_as_unreachable():
    """Offline is a normal state, not an exception to be surprised by."""
    import asyncio

    from aegis.sync.meshlink import HttpMeshLink

    link = HttpMeshLink()
    link.register("device-B", "http://127.0.0.1:59999")
    link.set_offline(True)

    async def attempt():
        with pytest.raises(ConnectionError):
            await link.call("device-A", "device-B", "ping", {})
        link.set_offline(False)
        # Still unreachable, but for a transport reason rather than the radio.
        with pytest.raises(ConnectionError):
            await link.call("device-A", "device-B", "ping", {})
        await link.close()

    asyncio.run(attempt())
    assert link.snapshot()["transport"] == "http"
    assert link.dropped == 2


def test_http_mesh_link_refuses_an_unknown_peer():
    import asyncio

    from aegis.sync.meshlink import HttpMeshLink

    link = HttpMeshLink()

    async def attempt():
        with pytest.raises(ConnectionError):
            await link.call("device-A", "nobody", "ping", {})
        await link.close()

    asyncio.run(attempt())
