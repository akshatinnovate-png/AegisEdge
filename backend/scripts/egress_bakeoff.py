"""If the link dies mid-sync, what got through?

    python3 scripts/egress_bakeoff.py --points 400 --budget 16384

A durable queue means nothing is lost. It does not mean everything arrives:
on an intermittent link the part that lands before the link drops is the part
that was worth sending, and the order decides which part that is.

This builds a realistic queue on a real node — memories ingested, some of them
actually retrieved, some deleted, some superseded — then cuts the link after a
fixed number of bytes and asks the same question of two orderings:

    FIFO    the order the writes happened in
    value   value per byte, obligations first, nothing starving

Both run against the same queue, the same budget and the same starvation rule,
so the only variable is the ordering. The verdict at the bottom is computed
from the two columns.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aegis.config import Settings                                # noqa: E402
from aegis.node import EdgeNode                                  # noqa: E402
from aegis.sync.crdt import OpKind                               # noqa: E402
from aegis.sync.egress import EgressPlanner                      # noqa: E402

O, R, B, D, G, Y = ("\033[38;5;208m", "\033[0m", "\033[1m", "\033[2m",
                    "\033[38;5;42m", "\033[38;5;220m")

WORDS = ("conveyor bearing vibration raceway gantry pump seal night shift line torque "
         "spindle coolant hydraulic valve actuator encoder relay inverter motor gearbox").split()


def synthetic(i: int, rng: random.Random) -> str:
    return f"observation {i}: " + " ".join(rng.sample(WORDS, 8))


def delivered(planner: EgressPlanner, ops, *, order, budget, points, divergent):
    """What lands, and what it was worth, if the link stops after `budget` bytes."""
    plan = planner.plan(ops, budget_bytes=budget, points=points, divergent=divergent,
                        order=order, link="metered")
    by_id = {a.op.op_id: a for a in plan.assessments}
    sent = [by_id[op.op_id] for op in plan.send if op.op_id in by_id]
    return {
        "ops": len(sent),
        "bytes": plan.planned_bytes,
        "value": sum(a.value for a in sent),
        "obligations": sum(1 for a in sent if a.op.kind is OpKind.DELETE),
        "redundant_suppressed": len(plan.redundant),
        "redundant_bytes": plan.redundant_bytes,
        "deferred": len(plan.deferred),
    }


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=400)
    parser.add_argument("--budget", type=int, default=16 * 1024,
                        help="bytes the link carries before it drops")
    parser.add_argument("--json", type=str, default="")
    args = parser.parse_args()

    node = EdgeNode(Settings())
    await node.start()
    quota = node.tenants.tenants["default"].quota
    quota.max_points = quota.max_ingest_per_minute = 10_000_000
    quota.max_qps, quota.max_bytes = 1e9, 1 << 40

    rng = random.Random(0)
    ids = []
    for i in range(args.points):
        point = await node.remember(synthetic(i, rng), "episodic")
        ids.append(point.id)

    # A realistic queue is not uniform. Some memories get read, a few get
    # deleted, a few get written twice — and which of those the link has time
    # for is the entire question.
    for _ in range(max(args.points // 8, 8)):
        await node.pipeline.search(" ".join(rng.sample(WORDS, 4)), k=5, collection="episodic")
    for point_id in rng.sample(ids, max(args.points // 40, 4)):
        point = node.store.points.get(point_id)
        if point is None:
            continue
        # The same two calls the delete route makes: the tombstone has to be
        # queued for egress, or the fleet never learns the memory is gone.
        node.sync.record_local(point, OpKind.DELETE)
        node.store.delete(point_id)
    for point_id in rng.sample(ids, max(args.points // 20, 6)):
        point = node.store.points.get(point_id)
        if point is not None:
            node.sync.record_local(point, OpKind.UPSERT)         # a second write, same point

    queued = list(node.sync.queue.pending)
    if not queued:
        print(f"{Y}the queue is empty — nothing to schedule{R}")
        await node.stop()
        return 1

    planner = EgressPlanner(node.settings.node_id)
    points = node.store.points
    divergent = node.sync._divergent_point_ids()
    total_bytes = sum(planner.size_of(op) for op in queued)

    fifo = delivered(planner, queued, order="fifo", budget=args.budget,
                     points=points, divergent=divergent)
    value = delivered(planner, queued, order="value", budget=args.budget,
                      points=points, divergent=divergent)
    whole = delivered(planner, queued, order="value", budget=0,
                      points=points, divergent=divergent)

    print(f"{O}{B}egress bake-off{R}  {len(queued):,} queued operations, "
          f"{total_bytes / 1024:.1f} KB if every one were sent")
    print(f"{D}the link carries {args.budget / 1024:.0f} KB and then drops{R}\n")

    print(f"{B}{'':<24}{'fifo':>12}{'value-first':>14}{R}")
    rows = [
        ("operations landed", fifo["ops"], value["ops"], "{:d}"),
        ("bytes used", fifo["bytes"], value["bytes"], "{:d}"),
        ("value landed", fifo["value"], value["value"], "{:.1f}"),
        ("deletes landed", fifo["obligations"], value["obligations"], "{:d}"),
    ]
    for name, left, right, fmt in rows:
        better = right >= left
        colour = G if better else Y
        print(f"{name:<24}{fmt.format(left):>12}{colour}{fmt.format(right):>14}{R}")

    share_fifo = fifo["value"] / whole["value"] if whole["value"] else 0.0
    share_value = value["value"] / whole["value"] if whole["value"] else 0.0
    total_deletes = whole["obligations"]
    print()
    print(f"{'share of all value':<24}{share_fifo:>12.1%}{G}{share_value:>14.1%}{R}"
          f"{D}   of what the whole queue is worth{R}")
    print(f"{'deletes outstanding':<24}{total_deletes - fifo['obligations']:>12d}"
          f"{G}{total_deletes - value['obligations']:>14d}{R}"
          f"{D}   still held on the edge after the drop{R}")
    print()
    print(f"{'redundant, not sent':<24}{value['redundant_suppressed']:>26d}"
          f"{D}   superseded before they left; both orderings gain this{R}")
    print(f"{'bytes never spent':<24}{value['redundant_bytes'] / 1024:>25.1f}K"
          f"{D}   FIFO would have paid full price for no-ops{R}")

    gain = (share_value - share_fifo)
    verdict = (f"value-first landed {gain:.0%} more of the queue's worth in the same bytes"
               if gain > 0.01 else
               "the two orderings are within a point of each other on this queue")
    if value["obligations"] > fifo["obligations"]:
        verdict += (f", and {value['obligations'] - fifo['obligations']} more delete(s) "
                    f"reached the cloud")
    print(f"\n{G}{B}verdict{R} {G}{verdict}{R}")
    print(f"{D}derived from the two columns above, on one queue and one budget — "
          f"a different drop point gives a different number, which is why the budget "
          f"is an argument{R}")

    payload = {"queued": len(queued), "total_bytes": total_bytes, "budget": args.budget,
               "fifo": fifo, "value": value, "whole": whole,
               "share_fifo": round(share_fifo, 4), "share_value": round(share_value, 4),
               "verdict": verdict}
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"{D}wrote {args.json}{R}")
    await node.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
