"""Peer-to-peer mesh: membership and anti-entropy rounds."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from pydantic import BaseModel, Field

from ..core.tenancy import Scope
from ..node import EdgeNode
from .deps import get_node
from .security import Principal, requires
from .models import PeerRequest


class ExchangeRequest(BaseModel):
    """One mesh RPC, arriving from a peer device."""

    sender: str
    method: str
    payload: dict = Field(default_factory=dict)

router = APIRouter(prefix="/api/v1/mesh", tags=["mesh"])


@router.get("/status")
async def status(node: EdgeNode = Depends(get_node)) -> dict:
    return node.mesh.snapshot()


@router.post("/peers")
async def add_peer(body: PeerRequest, node: EdgeNode = Depends(get_node)) -> dict:
    peer = node.mesh.add_peer(body.node_id, body.endpoint)
    return {"added": peer.as_dict(), "peers": len(node.mesh.peers)}


@router.delete("/peers/{node_id}")
async def drop_peer(node_id: str, node: EdgeNode = Depends(get_node)) -> dict:
    if node_id not in node.mesh.peers:
        raise HTTPException(status_code=404, detail="no such peer")
    node.mesh.peers.pop(node_id)
    return {"removed": node_id, "peers": len(node.mesh.peers)}


@router.post("/exchange")
async def exchange(body: ExchangeRequest, node: EdgeNode = Depends(get_node)) -> dict:
    """The receiving half of the mesh, when the peer is another device.

    This is deliberately thin. It hands the call straight to the same
    `GossipAgent` an in-process peer would reach, so both sides run one
    implementation of anti-entropy, IBLT reconciliation, causal delivery and
    policy filtering. A second code path for "remote" peers would be a second
    thing to keep correct, and the first one to drift.
    """
    try:
        return await node.mesh.handle(body.sender, body.method, body.payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/offline")
async def offline(on: bool = True, node: EdgeNode = Depends(get_node)) -> dict:
    """Pull this device's radio, or put it back.

    The mesh is device-to-device, so being offline here is not the same as
    losing the cloud uplink: the node keeps serving locally and keeps queueing
    its own operations, it simply cannot reach its peers.
    """
    link = node.mesh_link
    setter = getattr(link, "set_offline", None)
    if not callable(setter):
        raise HTTPException(status_code=409,
                            detail="this node's mesh link has no radio to pull")
    state = setter(on)
    node.bus.publish("mesh", "radio", offline=state,
                     message=("mesh radio <b>down</b> — peers unreachable, local memory "
                              "still authoritative" if state else
                              "mesh radio <b>up</b> — peers reachable again"))
    return {"offline": state, "link": link.snapshot()}


@router.post("/round")
async def round_now(peers: int = 2, node: EdgeNode = Depends(get_node)) -> dict:
    if not node.mesh.peers:
        raise HTTPException(status_code=409, detail="no peers known to this node")
    return {"rounds": await node.mesh.round(peers), "mesh": node.mesh.snapshot()}


class AttackRequest(BaseModel):
    """Mount one of the attacks the simulator mounts, against this live node."""

    kind: str = Field(description="tamper | impersonate | unsigned | honest")


@router.post("/attack")
async def attack(body: AttackRequest, node: EdgeNode = Depends(get_node),
                 _: Principal = Depends(requires(Scope.ADMIN))) -> dict:
    """Try to get a forged operation into this node, and report what happened.

    The deterministic simulator in `aegis/sim/` runs these against a simulated
    fleet, which is where they were found. This route runs the same four cases
    against the process actually serving this request, so the claim can be
    checked without taking anybody's word for the simulation — including mine.

    Nothing here is staged: each case builds a real `Operation`, encodes it with
    the real wire codec and hands it to the real `GossipAgent.handle`, which is
    the same entry point a peer device reaches over `/mesh/exchange`. The only
    thing the route knows that a peer does not is the attacker's key, and it
    uses it exactly as an attacker would — to sign in its own name while
    claiming somebody else's.

    Three things this route has to get right, none of which were right when it
    was first written:

    **It needs ADMIN.** A route whose whole purpose is to push forged
    operations at a node, and which teaches that node a key in the process,
    cannot be the one unauthenticated write in the router. It was.

    **Its probe identities are namespaced.** They used to be `edge-victim` and
    `edge-attacker`, learned into the node's real trust store. Trust-on-first-
    use is permanent by design, so any fleet that ever shipped a device by one
    of those names would find it refused forever, by a demo. The names now
    carry a `probe:` prefix and this node's own id, which no real device will
    present.

    **Its refusals are counted separately.** A probe that is indistinguishable
    from a real attack in the node's telemetry is worse than no probe: an
    operator watching `refused_forged` climb cannot tell a drill from an
    incident. The mesh snapshot reports probe counts on their own.
    """
    from ..sync.crdt import OpKind, Operation
    from ..sync.identity import DeviceIdentity

    mesh = node.mesh
    if mesh.identity is None:
        raise HTTPException(status_code=409, detail="this node is not signing operations")

    # The probe identities are created once per process and reused.
    #
    # Generating a fresh victim key on every request made the *second* call
    # fail — the node had already learned a key for that name and refused the
    # new one, which is trust-on-first-use doing precisely its job. Keeping
    # them stable is what an actual pair of devices would do; regenerating them
    # would be asking the node to accept an identity takeover in order to
    # demonstrate that it refuses identity takeovers.
    victim_id = f"probe:victim:{node.settings.node_id}"
    attacker_id = f"probe:attacker:{node.settings.node_id}"
    probes = getattr(node, "_attack_probes", None)
    if probes is None:
        probes = {"attacker": DeviceIdentity(attacker_id),
                  "victim": DeviceIdentity(victim_id)}
        node._attack_probes = probes
    attacker, author = probes["attacker"], probes["victim"]
    try:
        mesh.identity.learn(victim_id, author.public_key_b64,
                            certificate=probes.get("certificate"))
    except Exception as exc:
        # In enrolment mode the probe has no certificate from the fleet root
        # and cannot get one, so the node refuses it — correctly. That is an
        # answer, not a 500.
        raise HTTPException(
            status_code=409,
            detail=(f"this node requires enrolment, so it will not learn the probe "
                    f"identity: {exc}. The attacks it would mount are refused for "
                    f"that reason alone, which is the stronger result.")) from exc

    original = Operation(kind=OpKind.UPSERT, point_id="attack-probe",
                         device_id=victim_id,
                         body={"text": "dispatch the crew to grid 14",
                               "sensitivity": "internal"})
    original.sig = author.sign(original.as_dict())

    if body.kind == "honest":
        candidate, story = original, "an ordinary operation, signed by the device that wrote it"
    elif body.kind == "tamper":
        candidate = Operation.from_dict({**original.as_dict(),
                                         "body": {**original.body,
                                                  "text": "dispatch the crew to grid 41"}})
        candidate.sig = original.sig          # the relay cannot re-sign; it reuses
        story = "a relay forwarding somebody else's operation with the body rewritten"
    elif body.kind == "impersonate":
        candidate = Operation.from_dict({**original.as_dict(), "op_id": "",
                                         "body": {"text": "stand down",
                                                  "sensitivity": "internal"}})
        candidate.op_id = original.op_id + "X"
        candidate.sig = attacker.sign(candidate.as_dict())    # its own key, another's name
        story = "an operation written in another device's name, signed with the attacker's key"
    elif body.kind == "unsigned":
        candidate = Operation.from_dict({**original.as_dict(), "sig": ""})
        story = "an operation carrying no signature at all"
    else:
        raise HTTPException(status_code=400, detail=f"unknown attack {body.kind!r}")

    before = {"forged": mesh.refused_forged, "policy": mesh.refused_inbound,
              "verified": mesh.verified_ops}
    mesh.known.pop(candidate.op_id, None)
    frame = mesh.codec.encode([candidate.as_dict()])
    result = await mesh.handle(attacker_id, "push",
                               {"frame": frame, "keys": {attacker_id:
                                                         attacker.public_key_b64}})
    accepted = bool(result.get("accepted"))
    mesh.known.pop(candidate.op_id, None)
    if accepted:
        # The honest case is *supposed* to be accepted, which means it was
        # materialised: indexed, retrievable, and gossipable to every peer.
        # Rolling back only `mesh.known` left a synthetic memory in the store
        # behaving exactly like something a person had written. A drill that
        # leaves real state behind is not a drill.
        try:
            node.store.delete(candidate.point_id)
        except Exception:
            pass

    # Move what this probe caused out of the counters an operator reads as
    # evidence of a real attack, and into counters that say "drill".
    delta_forged = mesh.refused_forged - before["forged"]
    delta_policy = mesh.refused_inbound - before["policy"]
    delta_verified = mesh.verified_ops - before["verified"]
    mesh.refused_forged -= delta_forged
    mesh.refused_inbound -= delta_policy
    mesh.verified_ops -= delta_verified
    mesh.probe_refused += delta_forged + delta_policy
    mesh.probe_verified += delta_verified

    expected_accept = body.kind == "honest"
    return {
        "attack": body.kind,
        "what_it_did": story,
        "accepted": accepted,
        "expected": "accepted" if expected_accept else "refused",
        "correct": accepted == expected_accept,
        "refused_as_forged": delta_forged,
        "refused_by_policy": delta_policy,
        "verified": delta_verified,
        "note": ("Run against the live node, through the same handler a peer device "
                 "reaches. The honest case is here so a refusal of the other three "
                 "cannot be a node that simply refuses everything."),
    }
