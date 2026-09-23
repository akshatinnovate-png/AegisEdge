"""Index benchmark: recall and latency, measured not claimed.

    python3 scripts/bench.py [--n 4000] [--dim 96]

Compares exact BLAS scan, HNSW and IVF-PQ on the same corpus, on this
machine, and prints the crossover the cost model derived from it. The numbers
in the README come from this script; re-run it and disagree with them.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aegis.memory.ann import AdaptiveVectorIndex, CostModel, Strategy   # noqa: E402

O, B, D, R = "\033[38;5;208m", "\033[1m", "\033[2m", "\033[0m"


def corpus(n: int, dim: int, clustered: bool, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if clustered:
        centres = rng.normal(size=(max(n // 200, 6), dim))
        data = np.vstack([c + 0.3 * rng.normal(size=(n // len(centres) + 1, dim)) for c in centres])[:n]
    else:
        data = rng.normal(size=(n, dim))
    data = data.astype(np.float32)
    return data / np.linalg.norm(data, axis=1, keepdims=True)


def evaluate(index: AdaptiveVectorIndex, data: np.ndarray, queries: np.ndarray,
             k: int) -> dict[str, float]:
    truth = [set(np.argsort(-(data @ q))[:k].tolist()) for q in queries]
    started = time.perf_counter()
    found = [index.search(q, k) for q in queries]
    elapsed = (time.perf_counter() - started) / len(queries) * 1000
    hits = sum(len(t & {int(pid[1:]) for pid, _ in f}) for t, f in zip(truth, found))
    return {"recall": hits / (len(queries) * k), "ms_per_query": elapsed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4000)
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--queries", type=int, default=40)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    cost = CostModel().calibrate(dim=args.dim)
    report: dict[str, object] = {"cost_model": cost.as_dict(), "n": args.n, "dim": args.dim,
                                 "k": args.k, "results": {}}

    if not args.json:
        print(f"\n{O}{B}AegisEdge index benchmark{R}  n={args.n} dim={args.dim} k={args.k}")
        print(f"{D}{'─' * 78}{R}")
        print(f"  cost model: flat {cost.flat_ns_per_point:.1f} ns/point · "
              f"hop {cost.graph_ns_per_hop:.1f} ns · "
              f"hnsw≥{cost.hnsw_crossover:,} · ivf≥{cost.ivf_crossover:,}\n")
        print(f"  {'corpus':<11}{'strategy':<10}{'recall@k':>10}{'ms/query':>11}"
              f"{'build s':>10}{'bytes/pt':>11}")
        print(f"  {D}{'-' * 74}{R}")

    rng = np.random.default_rng(99)
    for label, clustered in (("clustered", True), ("uniform", False)):
        data = corpus(args.n, args.dim, clustered)
        queries = data[rng.choice(len(data), args.queries, replace=False)]
        for strategy in (Strategy.FLAT, Strategy.HNSW, Strategy.IVF_PQ):
            index = AdaptiveVectorIndex(args.dim, cost, label)
            started = time.perf_counter()
            for i, vector in enumerate(data):
                index.add(f"p{i}", vector)
            index.force(strategy)
            build = time.perf_counter() - started

            measured = evaluate(index, data, queries, args.k)
            snapshot = index.snapshot()
            per_point = args.dim * 4
            if strategy is Strategy.IVF_PQ and index.ivf is not None:
                per_point = index.ivf.pq.bytes_per_vector
            row = {**measured, "build_s": round(build, 2), "bytes_per_point": per_point}
            report["results"][f"{label}/{strategy.value}"] = row
            if not args.json:
                print(f"  {label:<11}{strategy.value:<10}{measured['recall']:>10.3f}"
                      f"{measured['ms_per_query']:>11.2f}{build:>10.1f}{per_point:>11}")

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"\n  {D}exact scan is the recall baseline by definition; the approximate{R}")
        print(f"  {D}strategies are judged against it on the same corpus.{R}\n")


if __name__ == "__main__":
    main()
