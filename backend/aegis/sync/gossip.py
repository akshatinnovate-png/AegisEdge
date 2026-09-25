"""Peer-to-peer mesh.

Two robots in a tunnel have no uplink, but they are ten metres apart. A
cloud-centric sync design leaves them both blind until someone drives out of
the tunnel; an edge-native one lets them reconcile directly.

Anti-entropy over IBLT digests finds what each peer is missing in one
exchange. Rumour mongering pushes genuinely new operations to a few random
peers immediately, and stops when they come back as duplicates — so a fact
learned by one device reaches the rest in log rounds without a broadcast
storm. Policy still applies: a memory that may not leave the device does not
leave it for a peer either.
"""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from ..core import determinism
from ..core.bus import EventBus
from ..core.metrics import METRICS
from .causal import CausalBuffer, VectorClock
from .identity import (DeviceIdentity, IdentityChanged,  # noqa: F401
                       UnknownDevice)
from .compression import WireCodec
from .crdt import Operation
from .iblt import IBLT


@dataclass
class Peer:
    node_id: str
    endpoint: str = ""
    last_seen: float = 0.0
    rtt_ms: float = 0.0
    trust: float = 0.75
    rounds: int = 0
    received: int = 0
    sent: int = 0
    failures: int = 0
    clock: dict[str, int] = field(default_factory=dict)

    @property
    def alive(self) -> bool:
        return self.last_seen > 0 and (determinism.now() - self.last_seen) < 90.0

    def as_dict(self) -> dict[str, Any]:
        return {"node_id": self.node_id, "endpoint": self.endpoint, "alive": self.alive,
                "last_seen_s": round(determinism.now() - self.last_seen, 1) if self.last_seen else None,
                "rtt_ms": round(self.rtt_ms, 1), "trust": round(self.trust, 2),
                "rounds": self.rounds, "received": self.received, "sent": self.sent,
                "failures": self.failures}


class MeshLink:
    """In-process peer link. A radio/BLE/mDNS transport swaps in behind this."""

    def __init__(self, latency_ms: float = 6.0, loss: float = 0.0) -> None:
        self.nodes: dict[str, "GossipAgent"] = {}
        self.latency_ms = latency_ms
        self.loss = loss
        self.partitions: set[tuple[str, str]] = set()
        self.messages = 0
        self.dropped = 0

    def join(self, agent: "GossipAgent") -> None:
        self.nodes[agent.node_id] = agent

    def partition(self, a: str, b: str, on: bool = True) -> None:
        key = tuple(sorted((a, b)))
        self.partitions.add(key) if on else self.partitions.discard(key)

    def reachable(self, a: str, b: str) -> bool:
        return tuple(sorted((a, b))) not in self.partitions

    def snapshot(self) -> dict[str, Any]:
        return {"transport": "memory", "nodes": sorted(self.nodes),
                "offline": False, "messages": self.messages, "dropped": self.dropped,
                "partitions": ["|".join(p) for p in self.partitions]}

    async def call(self, sender: str, target: str, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.messages += 1
        if target not in self.nodes or not self.reachable(sender, target):
            self.dropped += 1
            raise ConnectionError(f"{target} unreachable from {sender}")
        await asyncio.sleep(self.latency_ms * determinism.rng().uniform(0.7, 1.4) / 1000.0)
        if determinism.rng().random() < self.loss:
            self.dropped += 1
            raise ConnectionError("packet lost")
        return await self.nodes[target].handle(sender, method, payload)


class GossipAgent:
    """One device's view of the mesh."""

    FANOUT = 2                    # peers pushed per rumour round
    REDUNDANCY_LIMIT = 2          # stop gossiping a rumour after this many duplicate acks

    def __init__(self, node_id: str, link: MeshLink, bus: EventBus,
                 op_source: Callable[[], list[Operation]],
                 apply_op: Callable[[Operation], Awaitable[None]],
                 may_share: Callable[[Operation], bool] | None = None,
                 identity: "DeviceIdentity | None" = None) -> None:
        self.node_id = node_id
        self.link = link
        self.bus = bus
        self.op_source = op_source
        self.apply_op = apply_op
        self.may_share = may_share or (lambda _op: True)
        # Without an identity the mesh behaves exactly as it did before:
        # every operation is believed. That is the configuration the
        # simulator uses to reproduce the attack, so it has to stay
        # reachable rather than be quietly removed.
        self.identity = identity

        self.peers: dict[str, Peer] = {}
        self.known: dict[str, Operation] = {}
        self.clocks: dict[str, dict[str, int]] = {}      # op_id -> vector clock at creation
        self.causal = CausalBuffer(node_id)
        self.codec = WireCodec()
        self.rumours: dict[str, int] = {}
        self.rounds = 0
        self.ops_pulled = 0
        self.ops_pushed = 0
        self.withheld = 0
        self.refused_inbound = 0
        self.refused_forged = 0
        self.verified_ops = 0
        self.bytes_saved = 0
        link.join(self)

    # -- identity ---------------------------------------------------------

    def _key_bundle(self) -> dict[str, str]:
        """Every author key this node holds, to travel with the operations.

        An operation written on edge-A and relayed by edge-B arrives at edge-C,
        which may never have spoken to edge-A. Without this, edge-C cannot
        verify a signature it has no key for and would refuse a perfectly good
        operation — signing would break the very partition tolerance it is
        meant to protect.

        The cost is stated plainly: trust-on-first-use over a relay is only as
        good as the relay that carried the first key. What it still buys is the
        thing the simulator found — an established identity cannot be taken
        over, because a *change* to a known key is refused, and an operation
        cannot be edited in flight, because the relay cannot re-sign it.
        """
        if self.identity is None:
            return {}
        import base64
        from cryptography.hazmat.primitives import serialization
        return {device: base64.b64encode(key.public_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PublicFormat.Raw)).decode()
                for device, key in self.identity.known.items()}

    def _learn_keys(self, keys: dict[str, str] | None) -> None:
        if not keys or self.identity is None:
            return
        for device, public in keys.items():
            try:
                self.identity.learn(device, public)
            except IdentityChanged as exc:
                # Not a merge conflict. Somebody is claiming a name that is
                # already spoken for, and the honest holder of that name is
                # still out there using it.
                self.refused_forged += 1
                self.bus.publish("mesh", "identity_conflict", level="error",
                                 device=device, message=str(exc))

    def _admit(self, op: Operation, sender: str) -> bool:
        """Everything that has to be true before an operation is applied.

        Both inbound paths run through here. They used to differ: anti-entropy
        pulled operations straight into the store while the push path checked
        policy, which meant the same restricted memory was refused if it was
        offered and accepted if it was asked for. One door, checked once.
        """
        # Policy. Egress filtering on the sender is worth nothing against a
        # sender that skips it, so the receiver enforces its own rules. The
        # simulator found this at seed 5 and finds it there every time.
        if not self.may_share(op):
            self.refused_inbound += 1
            self.bus.publish(
                "mesh", "inbound_refused", level="warn", peer=sender, op_id=op.op_id,
                message=(f"refused an operation from <b>{sender}</b> that this "
                         f"node's own policy says may never leave a device"))
            return False

        if self.identity is None:
            return True

        # Integrity and authorship. A relay may route an operation; it may not
        # edit one, and it may not write one in somebody else's name.
        try:
            ok = self.identity.verify(op.as_dict(), op.sig)
        except UnknownDevice:
            ok = False
            reason = f"no key known for <b>{op.device_id}</b>"
        else:
            reason = (f"signature from <b>{op.device_id}</b> does not check out"
                      if not ok else "")
        if not ok:
            self.refused_forged += 1
            self.bus.publish(
                "mesh", "forgery_refused", level="error", peer=sender,
                op_id=op.op_id, author=op.device_id,
                message=f"refused an operation relayed by <b>{sender}</b>: {reason}")
            return False
        self.verified_ops += 1
        return True

    # -- membership -------------------------------------------------------

    def add_peer(self, node_id: str, endpoint: str = "") -> Peer:
        peer = self.peers.setdefault(node_id, Peer(node_id, endpoint))
        if endpoint and not peer.endpoint:
            peer.endpoint = endpoint
        # An in-process link resolves a peer by name; an HTTP one needs to be
        # told where it lives. Membership is the only place that knows.
        register = getattr(self.link, "register", None)
        if callable(register) and peer.endpoint:
            register(node_id, peer.endpoint)
        return peer

    def _sample(self, count: int) -> list[Peer]:
        """Prefer live peers; occasionally probe a dead one so partitions heal."""
        alive = [p for p in self.peers.values() if p.alive]
        dead = [p for p in self.peers.values() if not p.alive]
        chosen = determinism.rng().sample(alive, min(count, len(alive))) if alive else []
        if dead and (not chosen or determinism.rng().random() < 0.3):
            chosen.append(determinism.rng().choice(dead))
        return chosen

    # -- local state ------------------------------------------------------

    def note_local(self, op: Operation) -> dict[str, int] | None:
        """Stamp a locally created operation with this node's vector clock.

        The counter advances only for operations this device may actually
        broadcast. A vector clock describes the *broadcast sequence*, not the
        local write log: ticking it for a memory that policy keeps on-device
        would punch a permanent hole in that sequence, and every peer would
        stall forever waiting for a message that is never allowed to be sent.

        Operations that arrive without a clock are applied directly. The
        buffer preserves declared ordering; it does not invent it.
        """
        # Signed before the policy check: an operation that never leaves
        # this device today may be exported tomorrow, and an unsigned one
        # in the log is a gap in the record.
        if self.identity is not None and not op.sig and op.device_id == self.node_id:
            op.sig = self.identity.sign(op.as_dict())
        self.known[op.op_id] = op
        if not self.may_share(op):
            return None                       # local-only: never enters the sequence
        if op.op_id in self.clocks:
            return self.clocks[op.op_id]      # already broadcast: stamping twice would
                                              # advance the sequence past what peers saw
        clock = self.causal.local_event()
        self.clocks[op.op_id] = clock.pack()
        return clock.pack()

    def _shareable(self) -> dict[str, Operation]:
        """Everything this device is permitted to expose to a peer.

        The egress check is applied here, once, to the union of local and
        learned operations. Filtering only the local source and then merging
        `known` back in would re-admit exactly the operations policy had just
        refused — `known` exists for duplicate suppression, and must never be
        a second door out of the device.
        """
        candidates: dict[str, Operation] = {op.op_id: op for op in self.op_source()}
        candidates.update(self.known)
        shareable: dict[str, Operation] = {}
        for op_id, op in candidates.items():
            if self.may_share(op):
                shareable[op_id] = op
            else:
                self.withheld += 1
        return shareable

    # -- inbound ----------------------------------------------------------

    async def handle(self, sender: str, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        peer = self.add_peer(sender)
        peer.last_seen = determinism.now()
        peer.received += 1
        self._learn_keys(payload.get("keys"))

        if method == "ping":
            return {"node_id": self.node_id, "ops": len(self._shareable()),
                    "clock": self.causal.clock.pack(), "keys": self._key_bundle()}

        if method == "digest":
            mine = self._shareable()
            cells = int(payload.get("cells", 128))
            table = IBLT(cells).insert_many(mine.keys())
            return {"table": table.to_wire(), "ops": len(mine),
                    "keys": self._key_bundle()}

        if method == "fetch":
            mine = self._shareable()
            ids = [op_id for op_id in payload.get("op_ids", []) if op_id in mine]
            frame = self.codec.encode([mine[op_id].as_dict() for op_id in ids])
            return {"frame": frame, "keys": self._key_bundle(),
                    "clocks": {i: self.clocks[i] for i in ids if i in self.clocks}}

        if method == "push":
            ops = [Operation.from_dict(o) for o in self.codec.decode(payload["frame"])]
            clocks = payload.get("clocks", {})
            # Anti-entropy transfers a *set*; CRDT operations are commutative,
            # so bulk state transfer applies directly. Causal ordering is
            # enforced on the rumour path, where operations stream in one at a
            # time and a supersede can genuinely outrun what it supersedes.
            streaming = bool(payload.get("rumour"))
            fresh = 0
            for op in ops:
                if op.op_id in self.known:
                    continue
                if not self._admit(op, sender):
                    continue
                stamped = clocks.get(op.op_id) if streaming else None
                if stamped is None:
                    self.known[op.op_id] = op          # no declared dependencies: apply now
                    if op.op_id in clocks:
                        self.clocks.setdefault(op.op_id, clocks[op.op_id])
                        self.causal.clock = self.causal.clock.merge(
                            VectorClock.of(clocks[op.op_id]))
                    await self.apply_op(op)
                    fresh += 1
                    continue
                for delivered in self.causal.receive(op, VectorClock.of(stamped), sender):
                    self.known[delivered.op_id] = delivered
                    self.clocks.setdefault(delivered.op_id, stamped)
                    await self.apply_op(delivered)
                    fresh += 1
            self.ops_pulled += fresh
            if fresh:
                self.bus.publish("mesh", "received", peer=sender, ops=fresh,
                                 message=f"mesh: <b>{fresh}</b> new ops from <b>{sender}</b>")
            return {"accepted": fresh, "duplicates": len(ops) - fresh}

        raise ValueError(f"unknown method {method}")

    # -- outbound ---------------------------------------------------------

    async def anti_entropy(self, peer_id: str) -> dict[str, Any]:
        """One reconciliation round with a peer, via IBLT difference."""
        peer = self.add_peer(peer_id)
        mine = self._shareable()
        cells = IBLT.size_for(max(8, len(mine) // 8))
        started = determinism.monotonic()
        try:
            response = await self.link.call(self.node_id, peer_id, "digest", {"cells": cells, "keys": self._key_bundle()})
        except ConnectionError as exc:
            peer.failures += 1
            peer.trust = max(0.0, peer.trust - 0.05)
            return {"peer": peer_id, "error": str(exc)}

        self._learn_keys(response.get("keys"))
        peer.rtt_ms = (determinism.monotonic() - started) * 1000
        peer.last_seen = determinism.now()
        peer.rounds += 1
        self.rounds += 1

        local_table = IBLT(cells).insert_many(mine.keys())
        remote_table = IBLT.from_wire(response["table"])
        only_mine, only_theirs, complete = local_table.subtract(remote_table).decode()

        attempts = 0
        while not complete and attempts < 3:
            # the difference outran the table — widen and retry rather than
            # act on a partial answer
            cells *= 4
            attempts += 1
            response = await self.link.call(self.node_id, peer_id, "digest", {"cells": cells, "keys": self._key_bundle()})
            local_table = IBLT(cells).insert_many(mine.keys())
            remote_table = IBLT.from_wire(response["table"])
            only_mine, only_theirs, complete = local_table.subtract(remote_table).decode()

        pulled = pushed = 0
        if only_theirs:
            fetched = await self.link.call(self.node_id, peer_id, "fetch",
                                           {"op_ids": sorted(only_theirs),
                                            "keys": self._key_bundle()})
            ops = [Operation.from_dict(o) for o in self.codec.decode(fetched["frame"])]
            incoming_clocks = fetched.get("clocks", {})
            self._learn_keys(fetched.get("keys"))
            for op in ops:
                if op.op_id in self.known:
                    continue
                if not self._admit(op, peer_id):
                    continue
                self.known[op.op_id] = op
                if op.op_id in incoming_clocks:
                    self.clocks[op.op_id] = incoming_clocks[op.op_id]
                    self.causal.clock = self.causal.clock.merge(
                        VectorClock.of(incoming_clocks[op.op_id]))
                await self.apply_op(op)
                pulled += 1
            self.bytes_saved += max(0, fetched["frame"]["raw_bytes"] - fetched["frame"]["wire_bytes"])

        if only_mine:
            sending = [op_id for op_id in sorted(only_mine) if op_id in mine]
            frame = self.codec.encode([mine[op_id].as_dict() for op_id in sending])
            result = await self.link.call(self.node_id, peer_id, "push", {
                "frame": frame, "keys": self._key_bundle(),
                "clocks": {i: self.clocks[i] for i in sending if i in self.clocks},
            })
            pushed = int(result.get("accepted", 0))
            peer.sent += pushed
            self.ops_pushed += pushed

        self.ops_pulled += pulled
        peer.trust = min(1.0, peer.trust + 0.02)
        METRICS.incr("mesh.rounds")
        self.bus.publish(
            "mesh", "round", peer=peer_id, pulled=pulled, pushed=pushed,
            complete=complete, wire_cells=cells,
            message=(f"mesh round with <b>{peer_id}</b> · ↓{pulled} ↑{pushed} "
                     f"· {cells} IBLT cells"),
        )
        return {"peer": peer_id, "pulled": pulled, "pushed": pushed, "complete": complete,
                "cells": cells, "rtt_ms": round(peer.rtt_ms, 1)}

    async def rumour(self, op: Operation) -> int:
        """Push one hot operation to a few peers immediately."""
        if not self.may_share(op):
            self.withheld += 1
            return 0
        self.known.setdefault(op.op_id, op)
        delivered = 0
        for peer in self._sample(self.FANOUT):
            if self.rumours.get(op.op_id, 0) >= self.REDUNDANCY_LIMIT:
                break                              # everyone nearby already has it
            try:
                frame = self.codec.encode([op.as_dict()])
                result = await self.link.call(self.node_id, peer.node_id, "push", {
                    "frame": frame, "rumour": True, "keys": self._key_bundle(),
                    "clocks": {op.op_id: self.clocks[op.op_id]} if op.op_id in self.clocks else {},
                })
                if result.get("accepted"):
                    delivered += 1
                    peer.sent += 1
                else:
                    self.rumours[op.op_id] = self.rumours.get(op.op_id, 0) + 1
            except ConnectionError:
                peer.failures += 1
        self.ops_pushed += delivered
        return delivered

    async def round(self, peers: int = 2) -> list[dict[str, Any]]:
        sampled = self._sample(peers) or list(self.peers.values())[:peers]
        return [await self.anti_entropy(p.node_id) for p in sampled]

    def snapshot(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id, "peers": [p.as_dict() for p in self.peers.values()],
            "alive_peers": sum(1 for p in self.peers.values() if p.alive),
            "known_ops": len(self.known), "rounds": self.rounds,
            "pulled": self.ops_pulled, "pushed": self.ops_pushed,
            "withheld_by_policy": self.withheld,
            "refused_inbound": self.refused_inbound,
            "refused_forged": self.refused_forged,
            "verified_ops": self.verified_ops,
            "identity": (self.identity.snapshot() if self.identity is not None
                         else {"signing": "off — every operation is believed"}),
            "bytes_saved": self.bytes_saved,
            "causal": self.causal.snapshot(), "codec": self.codec.snapshot(),
            # The transport's own state, so a caller can tell a quiet mesh from
            # a pulled radio. Without it the two are indistinguishable from
            # outside, and anything rendering the link has to guess.
            "link": (self.link.snapshot() if hasattr(self.link, "snapshot")
                     else {"transport": type(self.link).__name__}),
        }
