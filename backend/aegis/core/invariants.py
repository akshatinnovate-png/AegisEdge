"""The simulator's invariants, running against the live node.

`aegis/sim/` checks seven properties after every step of every simulated
execution. Three thousand executions found no counterexample, which is a real
result and also a limited one: it says the code holds under the orderings the
simulator sampled, on the machine that ran it, with the faults it knows how to
inject. It says nothing about the device in somebody's hand.

So the same properties are checked here, continuously, against the running
node. The point is not that a violation is expected — it is that a property
worth testing is worth watching, and a system that can only detect its own
corruption in a test harness cannot detect it where it matters.

Two things make this affordable.

**A budget, not a sweep.** Verifying every signature in the log on every tick
is O(n) per tick and O(n^2) over a run, which would make the check the most
expensive thing the node does. Each invariant instead advances a rolling
cursor through its own subject and spends a fixed budget, so a long log is
covered over many ticks rather than all at once. Cost per tick is bounded by
the budget, not by how much the device has remembered.

**Locally checkable projections.** The simulator is omniscient: it knows what
every device holds and who really wrote what, so it can assert things a single
node cannot. A node sees only itself. What survives that restriction is
written here, and where an invariant is weaker than its simulated counterpart
the docstring says so rather than implying the node checks more than it does.

A violation is not an exception. It is counted, published on the bus at
`error`, and kept — the node keeps serving, because a node that halts on a
detected inconsistency converts a partial fault into a total one, and the
operator needs the evidence more than they need the process dead.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from . import determinism
from .clock import HLC


@dataclass
class Finding:
    """One property that did not hold, with enough context to act on."""

    invariant: str
    detail: str
    at: float = field(default_factory=determinism.now)

    def as_dict(self) -> dict[str, Any]:
        return {"invariant": self.invariant, "detail": self.detail, "at": round(self.at, 3)}


@dataclass
class LiveInvariant:
    name: str
    promise: str
    check: Callable[["InvariantMonitor"], str | None]
    # How far the local check falls short of the simulated one, stated rather
    # than left for a reader to discover.
    scope: str = ""
    checks: int = 0
    violations: int = 0
    cursor: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "promise": self.promise, "scope": self.scope,
                "assertions": self.checks, "violations": self.violations}


class InvariantMonitor:
    """Runs the live invariants against one node, on a budget."""

    BUDGET = 64               # subjects examined per invariant per tick
    KEEP_FINDINGS = 32

    def __init__(self, node: Any, budget: int | None = None) -> None:
        self.node = node
        self.budget = budget or self.BUDGET
        self.ticks = 0
        self.assertions = 0
        self.violations = 0
        self.findings: list[Finding] = []
        self.last_tick_ms = 0.0
        self.total_ms = 0.0
        self._last_hlc: HLC | None = None
        self._created: set[str] = set()
        self.invariants = _build()

    # -- the tick ---------------------------------------------------------

    def tick(self) -> list[Finding]:
        """Check every invariant once, each within its budget."""
        started = time.perf_counter()
        found: list[Finding] = []
        for invariant in self.invariants:
            try:
                detail = invariant.check(self)
            except Exception as exc:
                # A check that throws is itself a finding. Swallowing it would
                # leave the counter climbing while nothing was being checked,
                # which is worse than a violation because it looks like health.
                detail = f"the check itself failed: {type(exc).__name__}: {exc}"
            invariant.checks += 1
            self.assertions += 1
            if detail is None:
                continue
            invariant.violations += 1
            self.violations += 1
            finding = Finding(invariant.name, detail)
            found.append(finding)
            self.findings.append(finding)
            del self.findings[:-self.KEEP_FINDINGS]
            bus = getattr(self.node, "bus", None)
            if bus is not None:
                bus.publish("integrity", "invariant_violated", level="error",
                            invariant=invariant.name, detail=detail,
                            message=(f"invariant <b>{invariant.name}</b> does not hold: "
                                     f"{detail}"))
        self.ticks += 1
        self.last_tick_ms = (time.perf_counter() - started) * 1000
        self.total_ms += self.last_tick_ms
        return found

    # -- what the node can see --------------------------------------------

    def _oplog(self):
        return getattr(getattr(self.node, "sync", None), "oplog", None)

    def _mesh(self):
        return getattr(self.node, "mesh", None)

    def _slice(self, invariant: LiveInvariant, items: list) -> list:
        """The next `budget` subjects, wrapping round."""
        if not items:
            return []
        start = invariant.cursor % len(items)
        window = items[start:start + self.budget]
        if len(window) < self.budget and len(items) > self.budget:
            window += items[:self.budget - len(window)]
        invariant.cursor = (start + len(window)) % len(items)
        return window

    def note_created(self, op_id: str) -> None:
        """Remember an operation this device authored, for `origin-retains`."""
        self._created.add(op_id)
        if len(self._created) > 4096:
            self._created.pop()

    def snapshot(self) -> dict[str, Any]:
        return {
            "ticks": self.ticks,
            "assertions": self.assertions,
            "violations": self.violations,
            "budget_per_invariant": self.budget,
            "last_tick_ms": round(self.last_tick_ms, 3),
            "mean_tick_ms": round(self.total_ms / self.ticks, 3) if self.ticks else 0.0,
            "invariants": [i.as_dict() for i in self.invariants],
            "findings": [f.as_dict() for f in self.findings[-8:]],
            "claim": ("these are the properties aegis/sim checks after every simulated "
                      "step, checked here against the running node. A clean counter is "
                      "not a proof — it is this many assertions without a violation."),
        }


# -- the checks ------------------------------------------------------------
#
# Each returns None when the property holds, or a reason when it does not.

def _bodies_intact(m: InvariantMonitor) -> str | None:
    """Every signed operation still verifies against its author's key.

    The strongest of the local checks, and the one that would catch the attack
    the simulator found: an operation whose body was rewritten cannot carry a
    signature that checks out, so a node can detect the tampering by itself
    rather than needing to compare notes with the author.
    """
    mesh, log = m._mesh(), m._oplog()
    if mesh is None or log is None or mesh.identity is None:
        return None
    invariant = _by_name("bodies-intact")
    for op in m._slice(invariant, log.ops):
        if not op.sig:
            continue                      # unsigned deployments are a config, not a fault
        try:
            if not mesh.identity.verify(op.as_dict(), op.sig):
                return (f"operation {op.op_id} attributed to {op.device_id} does not "
                        f"verify against the key held for it")
        except Exception:
            continue                      # no key yet: `no-fabrication` owns that case
    return None


def _policy_holds(m: InvariantMonitor) -> str | None:
    """Nothing a device may not share is in the set it would share.

    Weaker than the simulated version, which can see every device. A node can
    only assert that *its own* egress set is clean — but that is the set it is
    about to hand a peer, so it is the one that decides whether a leak happens.
    """
    mesh = m._mesh()
    if mesh is None:
        return None
    invariant = _by_name("policy")
    try:
        shareable = mesh._shareable()
    except Exception:
        return None
    for op in m._slice(invariant, list(shareable.values())):
        if not mesh.may_share(op):
            return (f"operation {op.op_id} is in the set this device would hand a peer, "
                    f"and its own policy says it may never leave")
    return None


def _no_fabrication(m: InvariantMonitor) -> str | None:
    """Every operation held names a device, and a signed one names a known device."""
    log = m._oplog()
    mesh = m._mesh()
    if log is None:
        return None
    invariant = _by_name("no-fabrication")
    for op in m._slice(invariant, log.ops):
        if not op.device_id:
            return f"operation {op.op_id} is attributed to no device at all"
        if op.sig and mesh is not None and mesh.identity is not None:
            if op.device_id not in mesh.identity.known:
                return (f"operation {op.op_id} carries a signature from {op.device_id}, "
                        f"which this device has no key for")
    return None


def _no_duplicates(m: InvariantMonitor) -> str | None:
    """Applying an operation twice does not apply it twice.

    The op log is idempotent by construction — `seen` gates every append — so
    this asserts that construction still holds rather than trusting it.
    """
    log = m._oplog()
    if log is None:
        return None
    distinct = len({op.op_id for op in log.ops})
    if len(log.ops) != distinct:
        return (f"the log holds {len(log.ops)} operations under only {distinct} "
                f"distinct ids — an operation has been applied more than once")
    return None


def _clock_monotonic(m: InvariantMonitor) -> str | None:
    """This device's hybrid logical clock never runs backwards."""
    clock = getattr(m.node, "clock", None)
    if clock is None:
        return None
    current = clock.now()
    previous, m._last_hlc = m._last_hlc, current
    if previous is not None and not current.dominates(previous):
        return (f"the clock returned {current.pack()} after {previous.pack()} — "
                f"a later call produced an earlier stamp")
    return None


def _clock_not_poisoned(m: InvariantMonitor) -> str | None:
    """A peer claiming the future cannot carry this device there.

    The bound is generous — a device that has genuinely been offline for a
    week is not lying — and its purpose is to catch a clock that a peer has
    pushed years ahead, which is the attack the simulator mounts.
    """
    clock = getattr(m.node, "clock", None)
    if clock is None:
        return None
    ahead_ms = clock.now().wall_ms - int(determinism.now() * 1000)
    if ahead_ms > 86_400_000:
        return (f"this device's clock is {ahead_ms / 86_400_000:.1f} days ahead of its own "
                f"wall clock — a peer's timestamp claim has carried it there")
    return None


def _origin_retains(m: InvariantMonitor) -> str | None:
    """An operation this device wrote is still one this device holds."""
    log = m._oplog()
    if log is None or not m._created:
        return None
    missing = m._created - log.seen
    if missing:
        return (f"{len(missing)} operation(s) this device created are no longer in its "
                f"own log, the first being {sorted(missing)[0]}")
    return None


_REGISTRY: list[LiveInvariant] = []


def _by_name(name: str) -> LiveInvariant:
    return next(i for i in _REGISTRY if i.name == name)


def _build() -> list[LiveInvariant]:
    global _REGISTRY
    _REGISTRY = [
        LiveInvariant("bodies-intact", "an operation's content is what its creator wrote",
                      _bodies_intact,
                      "signed operations only; an unsigned deployment has nothing to check"),
        LiveInvariant("policy", "a restricted memory never leaves the device that made it",
                      _policy_holds,
                      "this device's own egress set; it cannot see what peers hold"),
        LiveInvariant("no-fabrication", "every operation held was created by some device",
                      _no_fabrication,
                      "attribution and key knowledge; it cannot prove the device existed"),
        LiveInvariant("no-duplicates", "delivering twice applies once", _no_duplicates),
        LiveInvariant("clock-monotonic", "a hybrid logical clock never runs backwards",
                      _clock_monotonic),
        LiveInvariant("clock-not-poisoned",
                      "a peer's timestamp claim cannot carry this device into the future",
                      _clock_not_poisoned, "bounded at one day of drift"),
        LiveInvariant("origin-retains", "a device keeps what it accepted",
                      _origin_retains,
                      "operations authored since this process started"),
    ]
    return _REGISTRY
