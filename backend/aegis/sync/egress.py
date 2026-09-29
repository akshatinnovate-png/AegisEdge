"""Which queued operations are worth the link we actually have.

Egress was a drain: while the queue had anything in it, send it, in the order
it was written. That is the right algorithm for a link that stays up. This
node's premise is a link that does not.

On an intermittent link the question is not *whether* to sync — the queue is
durable and everything goes eventually — but **what goes first**, because the
prefix that arrives before the link drops is the part that was worth sending.
FIFO answers that question with "whatever happened to be written first", which
is an answer, just not one about value.

Three decisions are made here, and each one is measurable rather than asserted.

**Redundant operations are not sent at all.** Two queued operations on the same
point mean the older one is dead on arrival: the cloud resolves last-writer-wins
by HLC, so applying it changes nothing. FIFO pays full price to transmit a
no-op. They are held, not dropped — an operation leaves the durable queue only
once the operation that supersedes it is acknowledged.

**The rest are ordered by value per byte.** The cost is the real encoded size.
The value is built from what the node already knows: a delete or a redaction is
an obligation rather than a convenience, a memory that gets retrieved is one
another device will want, a point inside a Merkle range the last cycle found
divergent closes a known gap.

**Nothing starves.** Value ordering, left alone, will never send a low-value
operation while higher-value ones keep arriving. Anything that has waited longer
than `STARVATION_S` is promoted ahead of the ordering entirely, so the wait is
bounded by a number rather than by hope.

Reordering egress is only sound because the merge at the far end is
order-independent: resolution is by HLC, not by arrival. That is a property of
the CRDT, and `test_the_merge_is_order_independent_which_is_what_lets_egress_reorder`
is there because the scheduler depends on it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..core import determinism
from .compression import WireCodec
from .crdt import Operation, OpKind


@dataclass(slots=True)
class Assessment:
    """One operation, priced and scored, with the reasons in words."""

    op: Operation
    value: float = 0.0
    bytes: int = 0
    reasons: list[str] = field(default_factory=list)
    starving: bool = False

    @property
    def ratio(self) -> float:
        return self.value / self.bytes if self.bytes else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"op_id": self.op.op_id, "point_id": self.op.point_id,
                "kind": self.op.kind.value, "value": round(self.value, 3),
                "bytes": self.bytes, "value_per_kb": round(self.ratio * 1024, 2),
                "reasons": list(self.reasons), "starving": self.starving}


@dataclass
class EgressPlan:
    send: list[Operation] = field(default_factory=list)
    redundant: list[Operation] = field(default_factory=list)
    deferred: list[Operation] = field(default_factory=list)
    assessments: list[Assessment] = field(default_factory=list)
    budget_bytes: int = 0
    planned_bytes: int = 0
    redundant_bytes: int = 0
    deferred_bytes: int = 0
    link: str = ""

    # A superseded operation may only be acknowledged once the operation that
    # supersedes it has been. Held here so the engine cannot ack it early by
    # accident, which would lose the write if the newer one never lands.
    releases: dict[str, list[str]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"link": self.link, "budget_bytes": self.budget_bytes,
                "send": len(self.send), "planned_bytes": self.planned_bytes,
                "redundant": len(self.redundant), "redundant_bytes": self.redundant_bytes,
                "deferred": len(self.deferred), "deferred_bytes": self.deferred_bytes,
                "starving": sum(1 for a in self.assessments if a.starving),
                "top": [a.as_dict() for a in self.assessments[:5]]}


class EgressPlanner:
    """Prices the queue against the link, and says why in words."""

    # A privacy obligation is not on the same axis as a useful update. The
    # cloud still holding what the edge deleted is a wrong state, not a stale
    # one, so a delete outranks any amount of usefulness.
    OBLIGATION = 40.0
    REDACTION = 24.0
    REPAIR = 12.0           # closes a divergence the last cycle actually found
    DEMAND = 6.0            # per log-unit of local retrievals
    PINNED = 8.0
    BASE = 1.0
    STARVATION_S = 120.0    # no operation waits longer than this in link time

    # Budgets per cycle, in bytes. HEALTHY is unbounded: there is no reason to
    # withhold on a good link, and ordering still decides the prefix that
    # survives if it drops mid-cycle.
    BUDGETS = {"healthy": 0, "metered": 96 * 1024, "degraded": 32 * 1024, "offline": -1}

    def __init__(self, node_id: str = "") -> None:
        self.node_id = node_id
        # A private codec: `WireCodec.encode` accumulates statistics, and
        # sizing the queue through the real one would report traffic that was
        # priced but never sent.
        self._ruler = WireCodec()
        self.planned = 0
        self.suppressed = 0
        self.suppressed_bytes = 0
        self.starved_promotions = 0
        self.last: EgressPlan | None = None

    # -- pricing ----------------------------------------------------------

    def released_bytes(self, plan: "EgressPlan", op_ids: Iterable[str]) -> int:
        """What the operations released on this cycle would have cost.

        The plan's `redundant_bytes` is the whole set. Only the ones whose
        superseding operation actually landed are released, so adding the total
        on a partial release counts bytes that were never saved — and counts
        the held remainder again on the next cycle.
        """
        wanted = set(op_ids)
        return sum(self.size_of(op) for op in plan.redundant if op.op_id in wanted)

    def size_of(self, op: Operation) -> int:
        """What this operation actually costs on the wire.

        Measured alone, so it is comparable across operations. A batch encodes
        smaller than the sum of its parts — the codec elides fields repeated
        within a frame — so this over-states the total and under-states nothing,
        which is the safe direction for a budget.
        """
        try:
            return int(self._ruler.encode([op.as_dict()])["wire_bytes"])
        except Exception:
            # Sizing must never be the reason an operation cannot be sent.
            return len(str(op.as_dict()).encode("utf-8", errors="ignore"))

    def value_of(self, op: Operation, *, point: Any = None,
                 divergent: set[str] | None = None) -> tuple[float, list[str]]:
        value, reasons = self.BASE, []
        body = op.body or {}

        if op.kind is OpKind.DELETE:
            value += self.OBLIGATION
            reasons.append("a delete the cloud has not applied is a wrong state, not a stale one")
        elif body.get("redacted") or body.get("metadata_only"):
            value += self.REDACTION
            reasons.append("carries a redaction the far end is still without")

        if divergent and op.point_id in divergent:
            value += self.REPAIR
            reasons.append("inside a range the last cycle found divergent")

        if point is not None:
            reads = float(getattr(point, "access_count", 0) or 0)
            if reads > 0:
                demand = self.DEMAND * math.log1p(reads)
                value += demand
                reasons.append(f"retrieved {int(reads)}x locally — the fleet will want it")
            if getattr(point, "pinned", False):
                value += self.PINNED
                reasons.append("pinned")
            confidence = float(getattr(point, "confidence", 1.0) or 1.0)
            if confidence < 0.6:
                value *= 0.6 + confidence
                reasons.append(f"low confidence ({confidence:.2f}) — worth less to the fleet")
        return value, reasons

    # -- planning ---------------------------------------------------------

    def plan(self, ops: Iterable[Operation], *, link: str = "healthy",
             points: dict[str, Any] | None = None, divergent: set[str] | None = None,
             budget_bytes: int | None = None, now: float | None = None,
             order: str = "value", record: bool = True) -> EgressPlan:
        """Decide what goes now. `order="fifo"` is the control, not a fallback.

        `record=False` prices the queue without touching the lifetime counters.
        `GET /sync/egress` shows an operator what a cycle *would* do and sends
        nothing; letting that inflate "operations suppressed" and "bytes not
        sent" would make a console poll look like traffic.
        """
        queued = list(ops)
        now = determinism.now() if now is None else now
        points = points or {}
        budget = self.BUDGETS.get(link, 0) if budget_bytes is None else budget_bytes
        plan = EgressPlan(budget_bytes=max(budget, 0), link=link)
        if not queued or budget < 0:
            plan.deferred = queued
            if record:
                self.last = plan
            return plan

        # 1. supersession. Newest per point by HLC wins; the rest are no-ops at
        #    the far end, and are held against that one's acknowledgement.
        newest: dict[str, Operation] = {}
        for op in queued:
            if op.kind is OpKind.DELETE:
                # A tombstone is never suppressed. Last-writer-wins makes an
                # older *upsert* a no-op once a newer one exists, which is what
                # makes suppression safe; a delete is not that. It is an OR-Set
                # tombstone, it carries information no upsert carries, and
                # deciding it was redundant would mean a memory someone asked
                # to be forgotten quietly never leaving the device it was
                # forgotten on.
                continue
            current = newest.get(op.point_id)
            if current is None or op.clock.dominates(current.clock):
                newest[op.point_id] = op
        survivors: list[Operation] = []
        for op in queued:
            keeper = newest.get(op.point_id)
            if (op.kind is not OpKind.DELETE and keeper is not None
                    and keeper.op_id != op.op_id):
                plan.redundant.append(op)
                plan.releases.setdefault(keeper.op_id, []).append(op.op_id)
            else:
                survivors.append(op)

        # 2. price what is left
        for op in survivors:
            value, reasons = self.value_of(op, point=points.get(op.point_id),
                                           divergent=divergent)
            assessment = Assessment(op=op, value=value, bytes=self.size_of(op),
                                    reasons=reasons)
            waited = now - float(op.ts or now)
            if waited >= self.STARVATION_S:
                assessment.starving = True
                assessment.reasons.append(
                    f"waited {waited:.0f}s — promoted ahead of the ordering")
            plan.assessments.append(assessment)

        plan.redundant_bytes = sum(self.size_of(op) for op in plan.redundant)

        # 3. order. Starving first, in age order, then by value per byte. FIFO
        #    keeps the same starvation rule so the comparison is about the
        #    ordering and nothing else.
        if order == "fifo":
            ordered = sorted(plan.assessments, key=lambda a: (not a.starving, a.op.ts))
        else:
            ordered = sorted(plan.assessments,
                             key=lambda a: (not a.starving,
                                            a.op.ts if a.starving else -a.ratio))
        if record:
            self.starved_promotions += sum(1 for a in ordered if a.starving)

        # 4. fill to the budget
        spent = 0
        for assessment in ordered:
            if budget and spent + assessment.bytes > budget and plan.send:
                plan.deferred.append(assessment.op)
                plan.deferred_bytes += assessment.bytes
                continue
            plan.send.append(assessment.op)
            spent += assessment.bytes
        plan.planned_bytes = spent
        plan.assessments = ordered

        if record:
            self.planned += len(plan.send)
            self.suppressed += len(plan.redundant)
            self.suppressed_bytes += plan.redundant_bytes
            self.last = plan
        return plan

    def snapshot(self) -> dict[str, Any]:
        return {
            "planned": self.planned,
            "suppressed_as_redundant": self.suppressed,
            "bytes_not_sent": self.suppressed_bytes,
            "starvation_promotions": self.starved_promotions,
            "starvation_ceiling_s": self.STARVATION_S,
            "budgets_bytes": dict(self.BUDGETS),
            "last": self.last.as_dict() if self.last else None,
        }
