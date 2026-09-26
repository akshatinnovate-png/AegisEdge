"""The smallest thing that reproduces the resident-set growth.

The open defect in §3 of the README. This is not a fix — it is a one-variable
reproducer, so that somebody can see it in two minutes rather than reading a
paragraph about it.

    python3 scripts/repro_growth.py            # both arms, ~6 min
    python3 scripts/repro_growth.py --arm seq  # just the one that leaks

Two runs, identical in every respect but one: whether 64 **sequential**
searches happen before the measured load. Both then warm with 8,000 concurrent
queries and measure the next 8,000, so neither measures start-up.

    without the sequential warm-up        7 B/query
    with it                           2,771 B/query

What that rules in and out is in the README. The short version: it is not any
pipeline stage — every rung of the degradation ladder behaves the same — it is
not the ONNX arena or the memory-pattern planner, and it does not go away if
the node's first inference is made concurrent, which was the obvious fix and
was measured and discarded.
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psutil                                                    # noqa: E402

from aegis.config import Settings                                # noqa: E402
from aegis.core.slo import Level                                 # noqa: E402
from aegis.node import EdgeNode                                  # noqa: E402

O, R, B, D = "\033[38;5;208m", "\033[0m", "\033[1m", "\033[2m"
WORDS = "conveyor bearing vibration raceway gantry pump seal night shift line torque".split()


def synthetic(i: int, rng: random.Random) -> str:
    return " ".join(rng.sample(WORDS, 4)) + f" {i % 17}"


# Eight thousand queries of start-up cost have to be paid before the slope is
# honest. Measured: the first four windows of two thousand run 8.8, 33.5, 5.5
# and 16.4 MB, and every window after that sits between 2.74 and 2.85 KB per
# query. A warm-up shorter than this does not merely add noise — it inverts the
# result, because the arm that has *not* done a sequential warm-up pays more of
# its start-up inside the measured window. Run at 2,400 it reports 12,029 B/query
# for the clean arm against 372 for the dirty one, which is the opposite
# conclusion, stated just as confidently.
WARM_ROUNDS = 1_000


async def arm(sequential_warmup: bool, batches: int) -> float:
    """One arm. Returns bytes of resident growth per query in the steady region."""
    node = EdgeNode(Settings())
    await node.start()
    proc = psutil.Process()
    rng = random.Random(0)
    for i in range(600):
        await node.remember(f"observation {i}: " + synthetic(i, rng), "episodic")
    queries = [synthetic(i, random.Random(i)) for i in range(128)]
    node.slo.override(Level.FULL)

    served = 0

    async def burn(rounds: int) -> None:
        nonlocal served
        for _ in range(rounds):
            await asyncio.gather(*(node.pipeline.search(queries[(served + i) % 128], k=5)
                                   for i in range(8)))
            served += 8

    if sequential_warmup:
        for query in queries[:64]:
            await node.pipeline.search(query, k=5)

    await burn(WARM_ROUNDS)                   # warm past the one-time start-up cost
    gc.collect()
    base, mark = proc.memory_info().rss, served
    await burn(batches)                       # the steady region, and the only thing measured
    gc.collect()
    grew, count = proc.memory_info().rss - base, served - mark
    await node.stop()
    print(f"  {'with' if sequential_warmup else 'without':>7} a sequential warm-up   "
          f"+{grew / 1e6:6.2f} MB over {count:,} queries = {B}{grew / count:7.1f} B/query{R}")
    return grew / count


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("both", "seq", "conc"), default="both")
    parser.add_argument("--batches", type=int, default=1000,
                        help="rounds of 8 concurrent queries in the MEASURED phase "
                             "(default 8,000 queries). The warm-up is fixed at "
                             f"{WARM_ROUNDS * 8:,} queries and is not adjustable, because "
                             "a shorter one inverts the result rather than blurring it.")
    args = parser.parse_args()

    print(f"{O}{B}resident-set growth{R}  one variable: 64 sequential searches "
          f"before the load")
    print(f"{D}both arms warm with {WARM_ROUNDS * 8:,} concurrent queries and measure "
          f"the next {args.batches * 8:,}{R}\n")

    rates = {}
    if args.arm in ("both", "conc"):
        rates["concurrent only"] = await arm(False, args.batches)
    if args.arm in ("both", "seq"):
        rates["sequential first"] = await arm(True, args.batches)

    if len(rates) == 2:
        clean, dirty = rates["concurrent only"], rates["sequential first"]
        ratio = dirty / max(clean, 1e-9)
        if ratio >= 2:
            print(f"\n  {B}{ratio:,.0f}x{R} more resident growth per query, from one "
                  f"difference before the measurement window opened.")
        else:
            # Never print a reassuring number for a result that did not come
            # out. If the arms agree, the run says so rather than rounding the
            # ratio to something that reads like a finding.
            print(f"\n  {O}the two arms did not separate{R} ({dirty:,.0f} against "
                  f"{clean:,.0f} B/query). On the hardware this was written on they "
                  f"differ by roughly 385x; if they do not here, say so rather than "
                  f"reporting the ratio.")
    print(f"\n{D}  Not a fix. A reproducer — see §3 of the README for what this rules "
          f"in and out.{R}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
