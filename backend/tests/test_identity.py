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


# -- enrolment: closing the first-contact gap ------------------------------

def _fleet_root():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    import base64
    root = Ed25519PrivateKey.generate()
    root_pub = base64.b64encode(root.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw)).decode()
    return root, root_pub


def test_an_enrolled_peer_is_accepted():
    root, root_pub = _fleet_root()
    peer = DeviceIdentity("edge-00")
    cert = DeviceIdentity.issue(root, "edge-00", peer.public_key_b64)
    node = DeviceIdentity("edge-01", root_public_b64=root_pub, require_enrolment=True)
    assert node.learn("edge-00", peer.public_key_b64, cert) is True


def test_a_stranger_at_first_contact_is_refused():
    """The gap trust-on-first-use cannot close, closed."""
    from aegis.sync.identity import NotEnrolled

    _root, root_pub = _fleet_root()
    attacker = DeviceIdentity("edge-00")            # never enrolled
    node = DeviceIdentity("edge-01", root_public_b64=root_pub, require_enrolment=True)
    with pytest.raises(NotEnrolled):
        node.learn("edge-00", attacker.public_key_b64, None)
    assert "edge-00" not in node.known


def test_the_same_stranger_is_believed_without_enrolment():
    """The control. Otherwise the test above only shows a node that refuses everything."""
    attacker = DeviceIdentity("edge-00")
    node = DeviceIdentity("edge-01")                # trust-on-first-use
    assert node.learn("edge-00", attacker.public_key_b64) is True


def test_a_certificate_cannot_be_reused_under_another_name():
    from aegis.sync.identity import NotEnrolled

    root, root_pub = _fleet_root()
    enrolled = DeviceIdentity("edge-00")
    cert = DeviceIdentity.issue(root, "edge-00", enrolled.public_key_b64)
    node = DeviceIdentity("edge-02", root_public_b64=root_pub, require_enrolment=True)
    with pytest.raises(NotEnrolled):                 # same certificate, different name
        node.learn("edge-99", enrolled.public_key_b64, cert)


def test_a_certificate_cannot_be_reused_with_another_key():
    from aegis.sync.identity import NotEnrolled

    root, root_pub = _fleet_root()
    enrolled, attacker = DeviceIdentity("edge-00"), DeviceIdentity("edge-00")
    cert = DeviceIdentity.issue(root, "edge-00", enrolled.public_key_b64)
    node = DeviceIdentity("edge-02", root_public_b64=root_pub, require_enrolment=True)
    with pytest.raises(NotEnrolled):                 # the name it was issued, a key it was not
        node.learn("edge-00", attacker.public_key_b64, cert)


def test_a_certificate_from_another_fleet_is_refused():
    from aegis.sync.identity import NotEnrolled

    other_root, _ = _fleet_root()
    _root, root_pub = _fleet_root()
    peer = DeviceIdentity("edge-00")
    cert = DeviceIdentity.issue(other_root, "edge-00", peer.public_key_b64)
    node = DeviceIdentity("edge-01", root_public_b64=root_pub, require_enrolment=True)
    with pytest.raises(NotEnrolled):
        node.learn("edge-00", peer.public_key_b64, cert)


def test_enrolment_still_refuses_a_key_change():
    """A certificate says whose a key is. It does not say an identity may have two."""
    root, root_pub = _fleet_root()
    first, second = DeviceIdentity("edge-00"), DeviceIdentity("edge-00")
    node = DeviceIdentity("edge-01", root_public_b64=root_pub, require_enrolment=True)
    node.learn("edge-00", first.public_key_b64,
               DeviceIdentity.issue(root, "edge-00", first.public_key_b64))
    with pytest.raises(IdentityChanged):
        node.learn("edge-00", second.public_key_b64,
                   DeviceIdentity.issue(root, "edge-00", second.public_key_b64))


def test_requiring_enrolment_without_a_root_is_refused_at_construction():
    """A node that required certificates it could not check would refuse its whole fleet."""
    with pytest.raises(ValueError):
        DeviceIdentity("edge-01", require_enrolment=True)


def test_the_snapshot_says_which_mode_it_is_in():
    _root, root_pub = _fleet_root()
    assert DeviceIdentity("edge-01").snapshot()["mode"] == "trust-on-first-use"
    strict = DeviceIdentity("edge-01", root_public_b64=root_pub, require_enrolment=True)
    assert strict.snapshot()["mode"] == "enrolment"
    assert "first contact is refused" in strict.snapshot()["trust"]


def test_an_enrolled_mesh_converges():
    """A defence that stops the fleet talking is not a defence."""
    root, root_pub = _fleet_root()
    bus, link = EventBus(), MeshLink(latency_ms=0.0)
    names = ["edge-00", "edge-01"]
    stores: dict[str, dict] = {n: {} for n in names}

    def applier(node):
        async def apply(op): stores[node][op.op_id] = op
        return apply

    identities = {}
    for n in names:
        ident = DeviceIdentity(n, root_public_b64=root_pub, require_enrolment=True)
        ident.certificate = DeviceIdentity.issue(root, n, ident.public_key_b64)
        identities[n] = ident

    agents = {n: GossipAgent(n, link, bus, op_source=lambda n=n: list(stores[n].values()),
                             apply_op=applier(n), identity=identities[n])
              for n in names}
    for a in agents.values():
        for other in names:
            if other != a.node_id:
                a.add_peer(other)

    op = _op("edge-00")
    stores["edge-00"][op.op_id] = op
    agents["edge-00"].note_local(op)
    asyncio.run(agents["edge-01"].anti_entropy("edge-00"))
    assert op.op_id in stores["edge-01"]


# -- arrays on the wire and under a signature ------------------------------

def test_an_array_in_a_body_signs_and_verifies_like_the_list_it_prints_as():
    """The op body shares the point's float32 array instead of copying it.

    Measured: 8,344 bytes per memory of pure duplication, on a log that is
    never trimmed. The array cannot go under a signature, so both the signer
    and the wire normalise it — and they must agree exactly, because a
    signature taken over one representation and checked against another fails
    as a *forgery*, which is the most misleading way for a serialisation bug
    to present.
    """
    import numpy as np

    from aegis.sync.identity import canonical

    author, peer = DeviceIdentity("edge-00"), DeviceIdentity("edge-01")
    peer.learn("edge-00", author.public_key_b64)
    vector = np.arange(8, dtype=np.float32) / 3.0

    as_array = _op("edge-00")
    as_array.body = {"text": "x", "sensitivity": "internal", "dense": vector}
    as_list = Operation.from_dict({**as_array.as_dict(), "body": {
        "text": "x", "sensitivity": "internal", "dense": vector.tolist()}})

    # The two must produce byte-identical signing input, or a node holding the
    # array and a peer holding the list would disagree about authenticity.
    assert canonical(as_array.as_dict()) == canonical(as_list.as_dict())

    signature = author.sign(as_array.as_dict())
    assert peer.verify(as_list.as_dict(), signature) is True


def test_an_array_body_survives_the_wire_and_still_verifies():
    import numpy as np

    author, peer = DeviceIdentity("edge-00"), DeviceIdentity("edge-01")
    peer.learn("edge-00", author.public_key_b64)
    op = _op("edge-00")
    op.body = {"text": "x", "sensitivity": "restricted",
               "dense": np.linspace(-1, 1, 16, dtype=np.float32)}
    op.sig = author.sign(op.as_dict())

    encoder, decoder = WireCodec(), WireCodec()
    back = Operation.from_dict(decoder.decode(encoder.encode([op.as_dict()]))[0])
    assert back.body["sensitivity"] == "restricted"          # the field that was dropped once
    assert peer.verify(back.as_dict(), back.sig) is True


def test_a_signed_operation_carrying_a_vector_survives_the_wire():
    """The defect this found: every real upsert was refused as a forgery.

    The codec quantized `dense` on the way out, so the vector that was signed
    was not the vector that arrived. No test had ever signed a body with a
    vector in it, so nothing caught it. The quantizing now happens where the
    operation is built, which is the only place it can happen without the
    signature and the wire disagreeing.
    """
    import numpy as np

    from aegis.sync.compression import quantize_vector

    author, peer = DeviceIdentity("edge-00"), DeviceIdentity("edge-01")
    peer.learn("edge-00", author.public_key_b64)
    op = _op("edge-00")
    op.body = {"text": "x", "sensitivity": "restricted",
               "dense_q": quantize_vector(np.linspace(-1, 1, 256, dtype=np.float32))}
    op.sig = author.sign(op.as_dict())

    encoder, decoder = WireCodec(), WireCodec()
    back = Operation.from_dict(decoder.decode(encoder.encode([op.as_dict()]))[0])
    assert back.body["sensitivity"] == "restricted"
    assert peer.verify(back.as_dict(), back.sig) is True


def test_a_quantized_body_is_a_fraction_of_a_list_body():
    """The memory this buys, on a log that is never trimmed."""
    import json

    import numpy as np

    from aegis.sync.compression import quantize_vector

    vector = np.linspace(-1, 1, 256, dtype=np.float32)
    as_list = len(json.dumps(vector.tolist()))
    as_codes = len(json.dumps(quantize_vector(vector)))
    assert as_codes * 3 < as_list, (as_codes, as_list)


def test_an_older_operation_carrying_a_plain_vector_is_still_read():
    """Operations already on disk and already in a peer's log keep working."""
    from aegis.sync.engine import _body_vector

    assert _body_vector({"dense": [0.25, 0.5]}) == [0.25, 0.5]
    assert _body_vector({}) is None


def test_the_attack_route_is_not_the_one_unauthenticated_write():
    """A route that pushes forged operations at a node needs ADMIN.

    It did not have it. Every other write in the API is scoped; this one, whose
    whole purpose is to hand a node something a peer should not be able to hand
    it, was reachable by anyone who could reach the port.
    """
    from fastapi.testclient import TestClient

    from aegis.api.security import principal
    from aegis.core.tenancy import Scope
    from aegis.main import app
    from aegis.api.security import Principal

    with TestClient(app) as client:
        # A principal with WRITE but not ADMIN must be turned away.
        app.dependency_overrides[principal] = lambda: Principal(
            "default", {Scope.READ, Scope.WRITE})
        try:
            refused = client.post("/api/v1/mesh/attack", json={"kind": "honest"})
        finally:
            app.dependency_overrides.pop(principal, None)
    assert refused.status_code == 403, refused.text


def test_the_probe_does_not_teach_the_node_a_real_device_name():
    """Trust-on-first-use is permanent, so a drill must not occupy a name.

    The probe identities were `edge-victim` and `edge-attacker`, learned into
    the node's real key store. Any fleet that ever shipped a device by one of
    those names would have found it refused forever, by a demo.
    """
    from fastapi.testclient import TestClient

    from aegis.main import app

    with TestClient(app) as client:
        client.post("/api/v1/mesh/attack", json={"kind": "honest"})
        known = client.get("/api/v1/mesh/status").json()["identity"]["known_devices"]

    assert not any(d in ("edge-victim", "edge-attacker") for d in known), known
    assert any(d.startswith("probe:") for d in known), known


def test_a_drill_is_not_counted_as_an_incident():
    """An operator watching `refused_forged` must be able to tell them apart."""
    from fastapi.testclient import TestClient

    from aegis.main import app

    with TestClient(app) as client:
        before = client.get("/api/v1/mesh/status").json()
        for kind in ("tamper", "impersonate", "unsigned"):
            assert client.post("/api/v1/mesh/attack", json={"kind": kind}).json()[
                "accepted"] is False
        after = client.get("/api/v1/mesh/status").json()

    assert after["refused_forged"] == before["refused_forged"], (
        "three refused forgeries from a drill moved the counter that reports real ones")
    assert after["probe_refused"] == before["probe_refused"] + 3


def test_a_private_key_is_never_briefly_world_readable(tmp_path):
    """`write_bytes` then `chmod` leaves a window. The window is enough.

    Checked by watching the mode from the moment the file appears rather than
    after the call returns, which is what a local attacker would be doing.
    """
    import os
    import threading

    from aegis.sync.identity import write_private_key

    target = tmp_path / "key.pem"
    temp = target.with_suffix(".pem.tmp")
    seen: list[int] = []
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            for path in (temp, target):
                try:
                    seen.append(os.stat(path).st_mode & 0o777)
                except FileNotFoundError:
                    pass

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    for _ in range(40):
        write_private_key(target, b"-----BEGIN PRIVATE KEY-----\nnot a real key\n")
    stop.set()
    watcher.join(timeout=5)

    assert seen, "the watcher never caught the file existing"
    assert all(mode == 0o600 for mode in seen), sorted(set(seen))


def test_a_failed_key_write_leaves_nothing_behind():
    """A half-written key that a later boot would load is worse than no key."""
    import pathlib

    from aegis.sync.identity import write_private_key

    with pytest.raises(Exception):
        write_private_key(pathlib.Path("/proc/nonexistent/key.pem"), b"x")
