"""Where does a memory's resident cost actually go?

    python3 scripts/memory_breakdown.py --count 6000

Ingest, then walk the node's own structures and attribute the growth. The
interesting line is the last one. Anything the walk cannot see is reported as
unattributed rather than assigned to whichever subsystem is convenient, and on
this machine that is most of it — which is the finding, not a failure of the
tool.

What has been isolated separately, each by removing exactly one thing and
measuring the difference:

    embedded Qdrant's upsert        2.9 KB/memory
    glibc arenas (MALLOC_ARENA_MAX) 4.1 KB/memory
    malloc_trim(0) hands back       1.6 KB/memory

None of those is the bulk either. The remainder is native, it grows with the
corpus, and it is not yet accounted for.
"""
from __future__ import annotations

import argparse
import asyncio
import ctypes
import gc
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                               # noqa: E402
import psutil                                                    # noqa: E402

from aegis.config import Settings                                # noqa: E402
from aegis.node import EdgeNode                                  # noqa: E402

O, R, B, D = "\033[38;5;208m", "\033[0m", "\033[1m", "\033[2m"
WORDS = ("conveyor bearing vibration raceway gantry pump seal night shift line torque "
         "spindle coolant hydraulic valve actuator encoder relay inverter motor").split()


def deep(obj: object, seen: set[int], depth: int = 0) -> int:
    """Bytes reachable from `obj` that nothing already counted holds.

    `seen` is shared across every subsystem measured, so an object two of them
    both reference is charged once, to whichever is walked first. That makes
    the individual rows approximate and the total sound, which is the right
    way round for the question being asked.
    """
    if id(obj) in seen or depth > 6:
        return 0
    seen.add(id(obj))
    if isinstance(obj, np.ndarray):
        return obj.nbytes + 128
    if isinstance(obj, (int, float, bool, type(None))):
        return 0
    total = sys.getsizeof(obj)
    try:
        if isinstance(obj, dict):
            total += sum(deep(k, seen, depth + 1) + deep(v, seen, depth + 1)
                         for k, v in list(obj.items()))
        elif isinstance(obj, (list, tuple, set, frozenset)):
            total += sum(deep(x, seen, depth + 1) for x in list(obj))
        elif hasattr(obj, "__slots__"):
            total += sum(deep(getattr(obj, s), seen, depth + 1)
                         for s in obj.__slots__ if hasattr(obj, s))
        elif hasattr(obj, "__dict__"):
            total += deep(vars(obj), seen, depth + 1)
    except Exception:
        pass                      # a structure that resists walking is not worth crashing for
    return total


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=6000)
    args = parser.parse_args()

    node = EdgeNode(Settings())
    await node.start()
    quota = node.tenants.tenants["default"].quota
    quota.max_points = quota.max_ingest_per_minute = 10_000_000
    quota.max_qps, quota.max_bytes = 1e9, 1 << 40

    proc, rng = psutil.Process(), random.Random(0)
    gc.collect()
    base = proc.memory_info().rss
    for i in range(args.count):
        await node.remember(f"observation {i}: " + " ".join(rng.sample(WORDS, 8)), "episodic")
    gc.collect()
    grew = proc.memory_info().rss - base

    print(f"{O}{B}resident grew {grew / 1e6:.1f} MB over {args.count:,} memories "
          f"= {grew / args.count / 1024:.1f} KB each{R}")
    if args.count < 4000:
        # Several rows are a fixed cost that does not grow with the corpus —
        # the embedder's geometry is about 5 MB whether the node holds one
        # memory or a million. Divided by a small count they look like a
        # per-memory cost and they are not, so the run says so rather than
        # letting a reader take the figure at face value.
        print(f"{O}  at {args.count:,} memories the fixed costs have not amortised: several rows"
              f" below are a constant divided by a small number. Use --count 6000 or more.{R}")
    print()

    subsystems = [
        ("store.points", node.store.points),
        ("sync.oplog.ops", node.sync.oplog.ops),
        ("sync.oplog index", (node.sync.oplog.seen, node.sync.oplog.heads)),
        ("mesh.known", node.mesh.known),
        ("merkle tree", node.sync.tree),
        ("vector index", getattr(node.store.store, "indexes", None)),
        ("sparse index", node.sparse),
        ("knowledge graph", node.graph),
        ("query understanding", node.understanding),
        ("embedder geometry", node.embedder),
        ("retrieval cache", node.pipeline.cache),
    ]
    seen: set[int] = set()
    attributed = 0
    for name, obj in subsystems:
        if obj is None:
            continue
        size = deep(obj, seen)
        attributed += size
        print(f"  {name:<24} {size / 1e6:>7.1f} MB   {size / args.count / 1024:>6.1f} KB/memory")

    rest = grew - attributed
    print(f"  {'─' * 24} {'─' * 7}")
    print(f"  {B}{'attributed':<24} {attributed / 1e6:>7.1f} MB   "
          f"{attributed / args.count / 1024:>6.1f} KB/memory{R}")
    print(f"  {O}{B}{'unattributed (native)':<24} {rest / 1e6:>7.1f} MB   "
          f"{rest / args.count / 1024:>6.1f} KB/memory{R}")

    try:
        before = proc.memory_info().rss
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        handed_back = before - proc.memory_info().rss
        print(f"\n{D}  malloc_trim(0) handed back {handed_back / 1e6:.1f} MB "
              f"({handed_back / args.count / 1024:.1f} KB/memory) — that was never data,"
              f"\n  it was free memory the allocator had not released. Run with "
              f"MALLOC_ARENA_MAX=2 to\n  bound how many arenas glibc opens; measured, that "
              f"is worth about 9% of the total.{R}")
    except Exception:
        pass

    await node.stop()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
