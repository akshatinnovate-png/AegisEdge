"""What signing is for, and what it is not for.

Each test here corresponds to something the deterministic simulator actually
found, or to a limit of the design that is better written down than discovered
later.
"""
import asyncio

import pytest

from aegis.core.bus import EventBus
from aegis.sync.compression import WireCodec
from aegis.sync.crdt import OpKind, Operation
from aegis.sync.gossip import GossipAgent, MeshLink
from aegis.sync.identity import (DeviceIdentity, IdentityChanged, UnknownDevice,
                                 canonical)


def _op(device: str, text: str = "observation", sensitivity: str = "internal") -> Operation:
    return Operation(kind=OpKind.UPSERT, point_id="p1", device_id=device,
                     body={"text": text, "sensitivity": sensitivity})


# -- the primitive ---------------------------------------------------------

def test_an_honest_operation_verifies():
    author, peer = DeviceIdentity("edge-00"), DeviceIdentity("edge-01")
    peer.learn("edge-00", author.public_key_b64)
    op = _op("edge-00")
    assert peer.verify(op.as_dict(), author.sign(op.as_dict())) is True


def test_a_rewritten_body_does_not_verify():
    """The relay attack: forward somebody else's operation with new content."""
    author, peer = DeviceIdentity("edge-00"), DeviceIdentity("edge-01")
    peer.learn("edge-00", author.public_key_b64)
    op = _op("edge-00", "dispatch to grid 14")
    signature = author.sign(op.as_dict())
    altered = _op("edge-00", "dispatch to grid 41")
    altered.op_id = op.op_id
    assert peer.verify(altered.as_dict(), signature) is False


def test_an_operation_signed_under_a_borrowed_name_does_not_verify():
    """Impersonation: the attacker can only sign with the key it holds."""
    victim, attacker, peer = (DeviceIdentity("edge-00"), DeviceIdentity("edge-99"),
                              DeviceIdentity("edge-01"))
    peer.learn("edge-00", victim.public_key_b64)
    spoof = _op("edge-00", "stand down")          # the lie is the device_id
    assert peer.verify(spoof.as_dict(), attacker.sign(spoof.as_dict())) is False


def test_an_unsigned_operation_does_not_verify():
    author, peer = DeviceIdentity("edge-00"), DeviceIdentity("edge-01")
    peer.learn("edge-00", author.public_key_b64)
    assert peer.verify(_op("edge-00").as_dict(), None) is False


def test_an_unknown_author_is_refused_rather_than_believed():
    peer = DeviceIdentity("edge-01")
    with pytest.raises(UnknownDevice):
        peer.verify(_op("edge-77").as_dict(), "irrelevant")


def test_a_known_identity_cannot_be_taken_over():
    """Trust on first use — and refusal on every use after."""
    victim, attacker, peer = (DeviceIdentity("edge-00"), DeviceIdentity("edge-99"),
                              DeviceIdentity("edge-01"))
    assert peer.learn("edge-00", victim.public_key_b64) is True
    assert peer.learn("edge-00", victim.public_key_b64) is True     # idempotent
    with pytest.raises(IdentityChanged):
        peer.learn("edge-00", attacker.public_key_b64)


def test_the_signature_is_not_part_of_what_it_covers():
    """Otherwise no signature could ever be checked against its own operation."""
    op = _op("edge-00")
    before = canonical(op.as_dict())
    op.sig = "anything at all"
    assert canonical(op.as_dict()) == before


def test_routing_fields_are_deliberately_not_signed():
    """`ts` is a relay's business; the content is not.

    Stated as a test so that anybody widening SIGNED_FIELDS has to decide to.
    """
    author, peer = DeviceIdentity("edge-00"), DeviceIdentity("edge-01")
    peer.learn("edge-00", author.public_key_b64)
    op = _op("edge-00")
    signature = author.sign(op.as_dict())
    op.ts += 3600.0
    assert peer.verify(op.as_dict(), signature) is True


# -- the wire --------------------------------------------------------------

def test_the_codec_preserves_every_body_field_across_frames():
    """The bug signing exposed.

    The encoder used to elide a repeated field against state that survived the
    frame, while the decoder started empty every time. A second frame carrying
    the same `sensitivity` value arrived with no `sensitivity` at all — and the
    receiving node decides what it may hold by reading exactly that field.
    """
    encoder = WireCodec()
    ops = [{"op_id": str(i), "kind": "upsert", "point_id": f"p{i}", "hlc": "",
            "device_id": "edge-00", "ts": 0.0, "sig": "",
            "body": {"text": f"t{i}", "sensitivity": "restricted", "collection": "mem"}}
           for i in range(4)]
    for _ in range(3):                                 # the second frame is where it broke
        assert WireCodec().decode(encoder.encode(ops)) == ops


def test_a_signature_survives_the_wire():
    author = DeviceIdentity("edge-00")
    peer = DeviceIdentity("edge-01")
    peer.learn("edge-00", author.public_key_b64)
    op = _op("edge-00", sensitivity="restricted")
    op.sig = author.sign(op.as_dict())
    encoder, decoder = WireCodec(), WireCodec()
    back = Operation.from_dict(decoder.decode(encoder.encode([op.as_dict()]))[0])
    assert peer.verify(back.as_dict(), back.sig) is True


# -- the mesh --------------------------------------------------------------

def _fleet(names, signed=True):
    bus, link = EventBus(), MeshLink(latency_ms=0.0)
    stores: dict[str, dict[str, Operation]] = {n: {} for n in names}

    def applier(node):
        async def apply(op): stores[node][op.op_id] = op
        return apply

    agents = {n: GossipAgent(n, link, bus,
                             op_source=lambda n=n: list(stores[n].values()),
                             apply_op=applier(n),
                             identity=DeviceIdentity(n) if signed else None)
              for n in names}
    for a in agents.values():
        for other in names:
            if other != a.node_id:
                a.add_peer(other)
    return agents, stores


def test_signed_operations_still_converge():
    """A defence that stops honest traffic is not a defence."""
    agents, stores = _fleet(["edge-00", "edge-01", "edge-02"])
    for i in range(6):
        op = _op("edge-00", f"observation {i}")
        op.point_id = f"p{i}"
        stores["edge-00"][op.op_id] = op
        agents["edge-00"].note_local(op)

    async def settle():
        for _ in range(4):
            for a in agents:
                for b in agents:
                    if a != b:
                        await agents[a].anti_entropy(b)
    asyncio.run(settle())
    assert all(len(s) == 6 for s in stores.values())


def test_a_third_party_key_travels_with_the_operation():
    """edge-02 must be able to verify an edge-00 operation relayed by edge-01."""
    agents, stores = _fleet(["edge-00", "edge-01", "edge-02"])
    op = _op("edge-00")
    stores["edge-00"][op.op_id] = op
    agents["edge-00"].note_local(op)

    async def relay():
        await agents["edge-01"].anti_entropy("edge-00")     # 01 learns 00's key
        await agents["edge-02"].anti_entropy("edge-01")     # 02 never met 00
    asyncio.run(relay())
    assert op.op_id in stores["edge-02"]
    assert "edge-00" in agents["edge-02"].identity.known


def test_a_relay_cannot_rewrite_an_operation_in_flight():
    """35 of 40 simulated executions found this before signing existed."""
    agents, stores = _fleet(["edge-00", "edge-01"])
    op = _op("edge-00", "dispatch to grid 14")
    stores["edge-00"][op.op_id] = op
    agents["edge-00"].note_local(op)
    agents["edge-01"].identity.learn("edge-00", agents["edge-00"].identity.public_key_b64)

    altered = _op("edge-00", "dispatch to grid 41")
    altered.op_id, altered.sig = op.op_id, op.sig
    frame = agents["edge-00"].codec.encode([altered.as_dict()])
    result = asyncio.run(agents["edge-01"].handle("edge-00", "push", {"frame": frame}))

    assert result["accepted"] == 0
    assert op.op_id not in stores["edge-01"]
    assert agents["edge-01"].refused_forged == 1


def test_a_peer_cannot_write_in_another_devices_name():
    agents, stores = _fleet(["edge-00", "edge-01", "edge-99"])
    agents["edge-01"].identity.learn("edge-00", agents["edge-00"].identity.public_key_b64)

    spoof = _op("edge-00", "stand down")
    spoof.sig = agents["edge-99"].identity.sign(spoof.as_dict())
    frame = agents["edge-99"].codec.encode([spoof.as_dict()])
    result = asyncio.run(agents["edge-01"].handle("edge-99", "push", {"frame": frame}))

    assert result["accepted"] == 0
    assert stores["edge-01"] == {}


def test_an_unsigned_mesh_still_accepts_the_attack():
    """The control. Without this the tests above prove only that nothing moves."""
    agents, stores = _fleet(["edge-00", "edge-01"], signed=False)
    op = _op("edge-00", "dispatch to grid 14")
    stores["edge-00"][op.op_id] = op
    agents["edge-00"].note_local(op)

    altered = _op("edge-00", "dispatch to grid 41")
    altered.op_id = op.op_id
    frame = agents["edge-00"].codec.encode([altered.as_dict()])
    asyncio.run(agents["edge-01"].handle("edge-00", "push", {"frame": frame}))
    assert stores["edge-01"][op.op_id].body["text"] == "dispatch to grid 41"


# -- the live node ---------------------------------------------------------

def test_the_attack_route_runs_the_attacks_against_the_running_node():
    """PROVE IT's sixth card, asserted rather than demonstrated.

    The honest case is the control: a node that refused everything would pass
    the other three for the wrong reason.
    """
    from fastapi.testclient import TestClient

    from aegis.main import app

    with TestClient(app) as client:
        outcomes = {}
        for kind in ("honest", "tamper", "impersonate", "unsigned"):
            response = client.post("/api/v1/mesh/attack", json={"kind": kind})
            assert response.status_code == 200, response.text
            outcomes[kind] = response.json()

    assert outcomes["honest"]["accepted"] is True
    for kind in ("tamper", "impersonate", "unsigned"):
        assert outcomes[kind]["accepted"] is False, outcomes[kind]
        assert outcomes[kind]["refused_as_forged"] == 1, outcomes[kind]
    assert all(o["correct"] for o in outcomes.values())
