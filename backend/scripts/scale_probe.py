"""Which term in the cost grows with the corpus?

    python3 scripts/scale_probe.py --to 10000 --step 2500

Ingest to a target, sampling ingest rate, query latency and resident set at
each step. The question is not whether the node is fast at two thousand
memories — §2 of the README already answers that — but what happens to each of
those as the corpus grows, because "edge" does not mean small.

The shipped tenant quota is lifted here on purpose. It is a real control and
it is not the subject: a probe that stopped at 600 writes a minute would
measure the quota rather than the system.
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psutil                                                    # noqa: E402

from aegis.config import Settings                                # noqa: E402
from aegis.node import EdgeNode                                  # noqa: E402

O, R, B, D = "\033[38;5;208m", "\033[0m", "\033[1m", "\033[2m"
WORDS = ("conveyor bearing vibration raceway gantry pump seal night shift line torque "
         "spindle coolant hydraulic valve actuator encoder relay inverter motor").split()


def synthetic(i: int, rng: random.Random) -> str:
    return f"observation {i}: " + " ".join(rng.sample(WORDS, 8))


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--to", type=int, default=10_000, dest="target")
    parser.add_argument("--step", type=int, default=2_500)
    args = parser.parse_args()

    node = EdgeNode(Settings())
    await node.start()
    quota = node.tenants.tenants["default"].quota
    quota.max_points = quota.max_ingest_per_minute = 10_000_000
    quota.max_qps, quota.max_bytes = 1e9, 1 << 40

    proc = psutil.Process()
    rng = random.Random(0)
    queries = [" ".join(rng.sample(WORDS, 4)) for _ in range(32)]

    print(f"{O}{B}scale probe{R}  to {args.target:,} points in steps of {args.step:,}")
    print(f"{D}the marginal cost of a memory is the slope of the last column, not its "
          f"ratio — most of the resident set at the first step is the model, not the corpus{R}\n")
    print(f"{'points':>8} {'docs/s':>9} {'ingest p95':>11} {'query p50':>10} "
          f"{'query p95':>10} {'resident':>10}")

    done, previous_rss = 0, None
    while done < args.target:
        latencies, started = [], time.perf_counter()
        for i in range(args.step):
            at = time.perf_counter()
            await node.remember(synthetic(done + i, rng), "episodic")
            latencies.append((time.perf_counter() - at) * 1000)
        elapsed = time.perf_counter() - started
        done += args.step

        queried = []
        for query in queries:
            at = time.perf_counter()
            await node.pipeline.search(query, k=5)
            queried.append((time.perf_counter() - at) * 1000)

        latencies.sort(); queried.sort()
        gc.collect()
        rss = proc.memory_info().rss
        marginal = ("" if previous_rss is None
                    else f"  {D}+{(rss - previous_rss) / args.step / 1024:.1f} KB/memory{R}")
        previous_rss = rss
        print(f"{done:>8,} {args.step / elapsed:>9.1f} "
              f"{latencies[int(len(latencies) * .95)]:>10.2f}m "
              f"{queried[len(queried) // 2]:>9.2f}m {queried[int(len(queried) * .95)]:>9.2f}m "
              f"{rss / 1e6:>9.1f}M{marginal}", flush=True)

    await node.stop()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
