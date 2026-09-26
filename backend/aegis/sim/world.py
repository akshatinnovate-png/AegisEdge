"""A seeded world where an entire fleet's history is a function of one integer.

Distributed bugs live in orderings nobody thinks to write a test for: a
partition that lands between the digest and the fetch, a clock that steps
backwards mid-merge, the same operation arriving twice from two directions, a
device that dies with an operation half-applied. Written by hand, those tests
cover the orderings their author imagined. Found by luck in production, they
are reproduced never and fixed by argument.

So the fleet is run inside `VirtualEnvironment`: time only moves when the
schedule says so, every coin comes from one seeded generator, and the whole
execution — every message, every fault, every interleaving — is determined by
the seed. Ten thousand simulated days cost ninety seconds of real time,
because sleeping is an addition rather than a wait.

After every single step, the invariants are checked. When one breaks, the run
stops and reports the seed and the step index, and `shrink` then removes
actions one at a time, re-running after each removal, until what is left is
the shortest sequence that still fails. A four-thousand-step failure becomes
six steps somebody can read.

What it is not: a proof. It samples the space of executions rather than
covering it, so a clean run means "no counterexample in this many attempts",
not "correct". That is a weaker claim than a model checker makes and a far
stronger one than a test suite makes, and the number of executions is reported
so a reader can judge it for themselves.
"""
from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Any, Callable

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ..core.bus import EventBus
from ..core.determinism import VirtualEnvironment, simulated
from ..core.clock import HLC
from ..sync.crdt import OpKind, Operation
from ..sync.gossip import GossipAgent, MeshLink
from ..sync.identity import DeviceIdentity

# What the scheduler can do to the world, and how often it chooses each.
ACTIONS = (
    ("write", 34),          # a device records a local observation
    ("gossip", 30),         # one peer runs anti-entropy against another
    ("partition", 8),       # two peers stop being able to reach each other
    ("heal", 8),            # ...and can again
    ("skew", 6),            # a device's clock steps, sometimes backwards
    ("crash", 5),           # a device loses everything not yet shared
    ("duplicate", 5),       # a message is delivered twice
    ("idle", 4),            # time passes and nothing happens
    # A peer that does not play by the rules. Every CRDT implementation
    # assumes honest participants, and a mesh of field devices is exactly
    # where that assumption is worth testing: one lost handset is one
    # dishonest peer.
    ("forge_clock", 4),     # claims a timestamp from the future, to win merges
    ("impersonate", 4),     # signs an operation as a device it is not
    ("tamper", 4),          # alters a body after the operation was made
)


@dataclass
class Violation:
    """An invariant that did not hold, and everything needed to see it again."""

    invariant: str
    detail: str
    seed: int
    step: int
    history: list[tuple[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"invariant": self.invariant, "detail": self.detail, "seed": self.seed,
                "step": self.step, "steps_to_reproduce": len(self.history),
                "history": [{"action": a, "args": list(b)} for a, b in self.history]}

    def __str__(self) -> str:
        return (f"{self.invariant} violated at seed {self.seed}, step {self.step}: "
                f"{self.detail}")


@dataclass
class Invariant:
    """A property that must hold after every step, whatever the schedule did."""

    name: str
    describe: str
    check: Callable[["Simulation"], str | None]      # returns a reason, or None


class Simulation:
    """A fleet of peers, a fault schedule, and the invariants in between."""

    def __init__(self, seed: int, peers: int = 5, world: VirtualEnvironment | None = None,
                 byzantine: int = 0, signed: bool = True) -> None:
        self.seed = int(seed)
        self.world = world
        self.peers = peers
        self.byzantine = byzantine
        # Signing is a switch rather than a constant so the simulator can
        # still reproduce the unsigned mesh on demand. A defence you cannot
        # turn off is a defence you can no longer demonstrate works.
        self.signed = signed
        self.bus = EventBus()
        self.link = MeshLink(latency_ms=0.0)
        self.names = [f"edge-{i:02d}" for i in range(peers)]
        self.stores: dict[str, dict[str, Operation]] = {n: {} for n in self.names}
        self.agents: dict[str, GossipAgent] = {}
        self.origin: dict[str, str] = {}          # op id -> the node that made it
        self.local_only: set[str] = set()         # op ids policy says never leave
        # Operations no honest device ever wrote. Convergence is a promise
        # about what the fleet actually observed, so a forgery that never
        # spreads is the system working, not a missing replica. Counting
        # these was the invariant reading a successful defence as a bug.
        self.fabricated: set[str] = set()
        # op id -> the device that stamped it in the future. A liar holding its
        # own lie is not poisoning anybody; the property is that it must not
        # spread, so the author is excluded and every other device is not.
        self.future_stamped: dict[str, str] = {}
        self.applied: dict[str, int] = {n: 0 for n in self.names}
        self.history: list[tuple[str, Any]] = []
        # Last seen vector clock per device, so `clock-monotonic` has a
        # previous value to compare against.
        self.clock_history: dict[str, dict[str, int]] = {}
        self.steps = 0
        self.messages = 0

        # Deterministic keys: the same seed must produce the same key
        # material, or a counterexample would not replay. Ed25519 takes any 32
        # bytes as a seed, so the simulation's own RNG supplies them.
        self.identities: dict[str, DeviceIdentity] = {}
        if self.signed:
            keyring = random.Random(self.seed ^ 0x5EED)
            for name in self.names:
                self.identities[name] = DeviceIdentity(
                    name, Ed25519PrivateKey.from_private_bytes(
                        bytes(keyring.getrandbits(8) for _ in range(32))))

        for name in self.names:
            self.agents[name] = GossipAgent(
                name, self.link, self.bus,
                op_source=lambda n=name: list(self.stores[n].values()),
                apply_op=self._applier(name),
                may_share=self._may_share,
                identity=self.identities.get(name))
        for agent in self.agents.values():
            for other in self.names:
                if other != agent.node_id:
                    agent.add_peer(other)

    # -- the world's rules -------------------------------------------------

    def _applier(self, target: str):
        async def apply(op: Operation) -> None:
            self.stores[target][op.op_id] = op
            self.applied[target] += 1
        return apply

    def _may_share(self, op: Operation) -> bool:
        """Policy: some memories never leave the device that made them."""
        return op.op_id not in self.local_only

    # -- what can happen ---------------------------------------------------

    async def _write(self, rng) -> tuple[str, Any]:
        node = rng.choice(self.names)
        restricted = rng.random() < 0.15
        op = Operation(kind=OpKind.UPSERT, point_id=f"p{self.steps}", device_id=node,
                       body={"text": f"observation {self.steps}",
                             "sensitivity": "restricted" if restricted else "internal"})
        self.stores[node][op.op_id] = op
        self.origin[op.op_id] = node
        if restricted:
            self.local_only.add(op.op_id)
        self.agents[node].note_local(op)
        return ("write", (node, op.op_id, restricted))

    async def _gossip(self, rng) -> tuple[str, Any]:
        a = rng.choice(self.names)
        b = rng.choice([n for n in self.names if n != a])
        try:
            await self.agents[a].anti_entropy(b)
            self.messages += 1
        except Exception:
            pass                      # an unreachable peer is a normal condition
        return ("gossip", (a, b))

    async def _partition(self, rng) -> tuple[str, Any]:
        a = rng.choice(self.names)
        b = rng.choice([n for n in self.names if n != a])
        self.link.partition(a, b, True)
        return ("partition", (a, b))

    async def _heal(self, rng) -> tuple[str, Any]:
        if not self.link.partitions:
            return ("heal", ())
        a, b = rng.choice(sorted(self.link.partitions))
        self.link.partition(a, b, False)
        return ("heal", (a, b))

    async def _skew(self, rng) -> tuple[str, Any]:
        seconds = rng.choice([-30.0, -5.0, -0.5, 0.5, 5.0, 30.0])
        if self.world is not None:
            self.world.skew(seconds)
        return ("skew", (seconds,))

    async def _crash(self, rng) -> tuple[str, Any]:
        """A device dies. Anything it never shared dies with it.

        This is the honest model of a crash on a device whose durable copy is
        its own log: operations already gossiped survive on their peers, and
        operations that never left are gone. A simulation that let a crash lose
        nothing would be testing a system nobody runs.
        """
        node = rng.choice(self.names)
        # "Elsewhere" has to mean everywhere the operation actually survives,
        # which includes each peer agent's own view and not just this
        # simulation's mirror of their stores. Deciding it from the mirror
        # alone made the very first run report a fabricated operation — a bug
        # in the model rather than in the system, which is the usual first
        # thing a simulator finds.
        elsewhere: set[str] = set()
        for other in self.names:
            if other == node:
                continue
            elsewhere |= set(self.stores[other])
            elsewhere |= set(getattr(self.agents[other], "known", {}))
        lost = [op_id for op_id in list(self.stores[node])
                if op_id not in elsewhere and self.origin.get(op_id) == node]
        for op_id in lost:
            del self.stores[node][op_id]
            self.agents[node].known.pop(op_id, None)
            self.origin.pop(op_id, None)
            self.local_only.discard(op_id)
        return ("crash", (node, len(lost)))

    async def _duplicate(self, rng) -> tuple[str, Any]:
        """The same operation delivered twice, which a network does routinely."""
        holders = [n for n in self.names if self.stores[n]]
        if not holders:
            return ("duplicate", ())
        node = rng.choice(holders)
        op = rng.choice(list(self.stores[node].values()))
        target = rng.choice([n for n in self.names if n != node])
        # Through the real wire path, frame and all, so a duplicate is
        # delivered exactly as the network would deliver it rather than by a
        # back door the production code never sees.
        frame = self.agents[node].codec.encode([op.as_dict()])
        await self.agents[target].handle(node, "push", {"frame": frame})
        return ("duplicate", (node, target, op.op_id))

    async def _forge_clock(self, rng) -> tuple[str, Any]:
        """A peer stamps an operation a century ahead so it wins every conflict.

        Last-writer-wins is only as trustworthy as the clock, and a hybrid
        logical clock merges what it is told. A peer that claims the future
        can make its version of a memory beat everybody's forever.
        """
        liar = rng.choice(self.names)
        target = rng.choice([n for n in self.names if n != liar])
        future = (self.world.now() if self.world else 0.0) + 3.15e9
        op = Operation(kind=OpKind.UPSERT, point_id=f"forged{self.steps}",
                       device_id=liar, hlc=f"{int(future * 1000)}.00000.{liar}",
                       body={"text": "forged", "sensitivity": "internal"})
        self.origin[op.op_id] = liar
        self.future_stamped[op.op_id] = liar
        self.stores[liar][op.op_id] = op
        # Signed honestly, with the liar's own key, under its own name. Nothing
        # about this operation is forged except the clock, so the signature is
        # valid and the only thing that can stop it is a bound on how far ahead
        # a peer is allowed to claim to be.
        self._attack_sign(liar, op, as_device=liar)
        frame = self.agents[liar].codec.encode([op.as_dict()])
        await self.agents[target].handle(liar, "push", self._attack_payload(liar, frame))
        return ("forge_clock", (liar, target, op.op_id))

    async def _impersonate(self, rng) -> tuple[str, Any]:
        """A peer sends an operation attributed to a device that did not make it."""
        liar = rng.choice(self.names)
        victim = rng.choice([n for n in self.names if n != liar])
        target = rng.choice([n for n in self.names if n not in (liar, victim)]) \
            if self.peers > 2 else victim
        op = Operation(kind=OpKind.UPSERT, point_id=f"spoof{self.steps}",
                       device_id=victim,          # the lie
                       body={"text": "attributed to a device that did not write it",
                             "sensitivity": "internal"})
        self.origin[op.op_id] = liar             # the truth, for the invariant
        self.fabricated.add(op.op_id)
        self.stores[liar][op.op_id] = op
        # The best the attacker can do: sign with the only key it holds, while
        # claiming the victim's name. The receiver checks the claimed name.
        self._attack_sign(liar, op, as_device=liar)
        frame = self.agents[liar].codec.encode([op.as_dict()])
        await self.agents[target].handle(liar, "push", self._attack_payload(liar, frame))
        return ("impersonate", (liar, victim, target, op.op_id))

    async def _tamper(self, rng) -> tuple[str, Any]:
        """A peer forwards somebody else's operation with the body rewritten."""
        holders = [n for n in self.names if self.stores[n]]
        if not holders:
            return ("tamper", ())
        liar = rng.choice(holders)
        original = rng.choice(list(self.stores[liar].values()))
        target = rng.choice([n for n in self.names if n != liar])
        altered = Operation(op_id=original.op_id, kind=original.kind,
                            point_id=original.point_id, hlc=original.hlc,
                            device_id=original.device_id,
                            body={**(original.body or {}), "text": "TAMPERED"})
        # The relay keeps the author's original signature and hopes nobody
        # checks it against the new body. Re-signing is not open to it: it does
        # not hold the author's key.
        altered.sig = original.sig
        frame = self.agents[liar].codec.encode([altered.as_dict()])
        await self.agents[target].handle(liar, "push", self._attack_payload(liar, frame))
        return ("tamper", (liar, target, original.op_id))

    def _attack_sign(self, liar: str, op: Operation, as_device: str) -> None:
        """Sign an attack operation with whatever key the attacker actually has."""
        identity = self.identities.get(liar)
        if identity is not None:
            op.sig = identity.sign({**op.as_dict(), "device_id": as_device})

    def _attack_payload(self, liar: str, frame: Any) -> dict[str, Any]:
        """An attacker offers its keys like anyone else; that is the point of TOFU."""
        return {"frame": frame, "keys": self.agents[liar]._key_bundle()}

    async def _idle(self, rng) -> tuple[str, Any]:
        if self.world is not None:
            self.world.advance(rng.uniform(0.1, 30.0))
        return ("idle", ())

    # -- invariants --------------------------------------------------------

    INVARIANTS: tuple[Invariant, ...] = ()        # populated below

    def check(self) -> Violation | None:
        for invariant in self.INVARIANTS:
            reason = invariant.check(self)
            if reason is not None:
                return Violation(invariant.name, reason, self.seed, self.steps,
                                 list(self.history))
        return None

    async def quiesce(self, rounds: int = 12) -> None:
        """Heal everything and gossip until it settles — then convergence is due."""
        self.link.partitions.clear()
        for _ in range(rounds):
            for name in self.names:
                for other in self.names:
                    if other != name:
                        try:
                            await self.agents[name].anti_entropy(other)
                        except Exception:
                            pass

    # -- the run -----------------------------------------------------------

    async def replay_step(self, action: str, args: tuple) -> tuple[str, Any]:
        """Perform one recorded action exactly, without consulting the generator.

        Shrinking depends on this. Re-rolling the dice for a replayed step
        would produce a different history and a different answer, which makes
        the shrinker's question — "does it still fail without this step?" —
        unanswerable.
        """
        if action == "write" and args:
            node, op_id, restricted = args
            op = Operation(kind=OpKind.UPSERT, point_id=f"p{self.steps}", device_id=node,
                           body={"text": f"observation {self.steps}",
                                 "sensitivity": "restricted" if restricted else "internal"},
                           op_id=op_id)
            self.stores[node][op.op_id] = op
            self.origin[op.op_id] = node
            if restricted:
                self.local_only.add(op.op_id)
            self.agents[node].note_local(op)
        elif action == "gossip" and args:
            a, b = args
            try:
                await self.agents[a].anti_entropy(b)
                self.messages += 1
            except Exception:
                pass
        elif action == "partition" and args:
            self.link.partition(args[0], args[1], True)
        elif action == "heal" and args:
            self.link.partition(args[0], args[1], False)
        elif action == "skew" and args:
            if self.world is not None:
                self.world.skew(args[0])
        elif action == "crash" and args:
            node = args[0]
            elsewhere: set[str] = set()
            for other in self.names:
                if other != node:
                    elsewhere |= set(self.stores[other])
                    elsewhere |= set(getattr(self.agents[other], "known", {}))
            for op_id in [o for o in list(self.stores[node])
                          if o not in elsewhere and self.origin.get(o) == node]:
                del self.stores[node][op_id]
                self.agents[node].known.pop(op_id, None)
                self.origin.pop(op_id, None)
                self.local_only.discard(op_id)
        elif action == "duplicate" and len(args) == 3:
            node, target, op_id = args
            op = self.stores.get(node, {}).get(op_id)
            if op is not None:
                frame = self.agents[node].codec.encode([op.as_dict()])
                await self.agents[target].handle(node, "push", {"frame": frame})
        elif action == "idle":
            if self.world is not None:
                self.world.advance(1.0)
        return (action, args)

    async def run(self, steps: int, replay: list[tuple[str, Any]] | None = None
                  ) -> Violation | None:
        rng = self.world.random if self.world else None
        handlers = {"write": self._write, "gossip": self._gossip,
                    "partition": self._partition, "heal": self._heal,
                    "skew": self._skew, "crash": self._crash,
                    "duplicate": self._duplicate, "idle": self._idle,
                    "forge_clock": self._forge_clock,
                    "impersonate": self._impersonate, "tamper": self._tamper}
        population = [a for a, _ in ACTIONS]
        weights = [w for _, w in ACTIONS]

        for index in range(steps):
            self.steps = index
            if replay is not None:
                if index >= len(replay):
                    break
                action, args = replay[index]
                record = await self.replay_step(action, tuple(args))
            else:
                action = rng.choices(population, weights=weights, k=1)[0]
                record = await handlers[action](rng)
            self.history.append(record)
            if self.world is not None:
                self.world.advance(0.05)
            violation = self.check()
            if violation is not None:
                return violation

        await self.quiesce()
        self.steps = len(self.history)
        return self.check_converged()

    def check_converged(self) -> Violation | None:
        """The property the whole design exists to provide."""
        # Operations the fleet refuses on purpose are not operations that
        # should converge. Forgeries never spread because no peer can verify
        # them; future-stamped operations never spread because every peer now
        # refuses an implausible clock. Counting either as a missing replica
        # reads a working defence as a fault.
        excluded = self.local_only | self.fabricated | set(self.future_stamped)
        shareable: dict[str, set[str]] = {
            n: {op_id for op_id in store if op_id not in excluded}
            for n, store in self.stores.items()}
        everything = set().union(*shareable.values()) if shareable else set()
        for name, held in shareable.items():
            missing = everything - held
            if missing:
                return Violation(
                    "convergence",
                    f"{name} is missing {len(missing)} shareable operation(s) after "
                    f"the partitions healed and gossip settled",
                    self.seed, self.steps, list(self.history))
        return None

    def snapshot(self) -> dict[str, Any]:
        return {"seed": self.seed, "peers": self.peers, "steps": len(self.history),
                "messages": self.messages,
                "operations": sum(len(s) for s in self.stores.values()),
                "restricted": len(self.local_only),
                "partitions": len(self.link.partitions)}


# -- the properties that must hold, whatever the schedule did ---------------

def _no_fabrication(sim: Simulation) -> str | None:
    """No node may hold an operation that no node ever created."""
    for name, store in sim.stores.items():
        for op_id in store:
            if op_id not in sim.origin:
                return f"{name} holds {op_id}, which no device created"
    return None


def _policy_holds(sim: Simulation) -> str | None:
    """A restricted memory never appears anywhere but the device that made it."""
    for op_id in sim.local_only:
        home = sim.origin.get(op_id)
        for name, store in sim.stores.items():
            if op_id in store and name != home:
                return (f"restricted operation {op_id} made on {home} is present "
                        f"on {name}")
    return None


def _origin_retains(sim: Simulation) -> str | None:
    """A device does not silently forget what it accepted, absent a crash."""
    for op_id, home in sim.origin.items():
        if op_id not in sim.stores[home]:
            return f"{home} lost {op_id}, which it created and did not crash away"
    return None


def _no_duplicates(sim: Simulation) -> str | None:
    """Applying an operation twice leaves one of it, not two."""
    for name, store in sim.stores.items():
        ids = list(store)
        if len(ids) != len(set(ids)):
            return f"{name} holds the same operation more than once"
    return None


def _clock_monotonic(sim: Simulation) -> str | None:
    """No device's vector clock ever goes backwards.

    This used to read `agent.causal.clock.last`, which does not exist —
    `VectorClock` has no such attribute, so the lookup returned None, the loop
    hit `continue` on every agent, and the invariant asserted nothing for three
    thousand executions. It was found by a review, not by a failure, which is
    the way a dead assertion is always found.

    What it checks now is real and local: every entry in a device's vector
    clock is monotone non-decreasing across steps. A counter that goes
    backwards means delivery ordering has been corrupted, and every causal
    guarantee above it is void.
    """
    for name, agent in sim.agents.items():
        current = agent.causal.clock.mapping
        previous = sim.clock_history.get(name)
        if previous is not None:
            for device, counter in previous.items():
                if current.get(device, 0) < counter:
                    return (f"{name}'s vector clock for {device} went backwards: "
                            f"{counter} then {current.get(device, 0)}")
        sim.clock_history[name] = current
    return None


def _clock_not_poisoned(sim: Simulation) -> str | None:
    """A peer's claim about the time cannot carry this device into the future.

    Last-writer-wins is only as trustworthy as the clock it compares. An
    operation stamped a century ahead dominates every honest write to the same
    point, forever — `HLC.dominates` compares `wall_ms` first and has no
    opinion about whether that number is plausible.

    So this asserts the plausibility that nothing else does: no operation an
    honest device holds may claim a time far beyond the world's own. An hour of
    genuine drift is fine and a century is not, which is the entire distinction
    the check exists to draw.
    """
    if sim.world is None:
        return None
    ceiling_ms = (sim.world.now() + 3600.0) * 1000.0
    for name, store in sim.stores.items():
        for op_id, op in store.items():
            if not op.hlc or sim.future_stamped.get(op_id) == name:
                continue          # a liar holding its own lie has poisoned nobody
            try:
                stamped = HLC.parse(op.hlc).wall_ms
            except Exception:
                return f"{name} holds {op_id} with an unparseable clock {op.hlc!r}"
            if stamped > ceiling_ms:
                ahead_days = (stamped - ceiling_ms) / 86_400_000.0
                return (f"{name} holds {op_id} stamped {ahead_days:,.0f} days beyond "
                        f"the world's clock — a peer's lie about the time, which "
                        f"last-writer-wins will honour forever")
    return None


def _bodies_intact(sim: Simulation) -> str | None:
    """An operation's content is what its creator wrote, not what a relay said."""
    for op_id, home in sim.origin.items():
        original = sim.stores.get(home, {}).get(op_id)
        if original is None:
            continue
        for name, store in sim.stores.items():
            held = store.get(op_id)
            if held is None or name == home:
                continue
            if (held.body or {}).get("text") != (original.body or {}).get("text"):
                return (f"{name} holds {op_id} with content that differs from what "
                        f"{home} wrote — a relay altered it in flight")
    return None


Simulation.INVARIANTS = (
    Invariant("no-fabrication", "every operation held was created by some device",
              _no_fabrication),
    Invariant("policy", "a restricted memory never leaves the device that made it",
              _policy_holds),
    Invariant("origin-retains", "a device keeps what it accepted unless it crashed",
              _origin_retains),
    Invariant("no-duplicates", "delivering twice applies once", _no_duplicates),
    Invariant("clock-monotonic", "a hybrid logical clock never runs backwards",
              _clock_monotonic),
    Invariant("clock-not-poisoned",
              "a peer's timestamp claim cannot carry this device into the future",
              _clock_not_poisoned),
    Invariant("bodies-intact",
              "an operation's content is what its creator wrote", _bodies_intact),
)


def run_seed(seed: int, steps: int = 300, peers: int = 5,
             replay: list[tuple[str, Any]] | None = None,
             signed: bool = True) -> tuple[Violation | None, dict]:
    """One complete execution. Same seed, same history, every time."""
    with simulated(seed) as world:
        sim = Simulation(seed, peers=peers, world=world, signed=signed)
        violation = asyncio.run(sim.run(steps, replay=replay))
        return violation, {**sim.snapshot(), **world.snapshot()}
