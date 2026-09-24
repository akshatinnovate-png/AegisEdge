"""Is the cost model's refusal to switch index strategy actually correct?

Every scale the battery tests comes back `flat`, and a reader is entitled to
suspect the adaptive index simply never fires. The claim in its defence — that
a graph traversal in the Python interpreter loses to one BLAS call over a
contiguous matrix until the corpus is very large — is a claim, so it gets
measured rather than asserted.

This forces each strategy onto the same corpus and reports what the choice
actually costs: latency at several percentiles, recall against exhaustive
search, and the time to build the structure in the first place.
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


def percentiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    def at(q: float) -> float:
        return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 3)
    return {"p50": at(0.5), "p95": at(0.95), "p99": at(0.99),
            "mean": round(sum(values) / len(values), 3)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, nargs="+", default=[5_000, 20_000])
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--out", default="../testlogs/strategy-bakeoff.json")
    args = parser.parse_args()

    cost = CostModel().calibrate(dim=args.dim, sample=512)
    report = {"cost_model": cost.as_dict(), "dim": args.dim, "k": args.k,
              "corpora": []}
    print(f"calibrated crossover: hnsw at {cost.hnsw_crossover:,} points, "
          f"ivf at {cost.ivf_crossover:,}\n")

    for count in args.points:
        rng = np.random.default_rng(count)
        # Clustered, not uniform: on uniform noise every ANN structure looks
        # bad, and no real corpus is uniform.
        centres = rng.normal(size=(40, args.dim))
        centres /= np.linalg.norm(centres, axis=1, keepdims=True)
        data = centres[rng.integers(0, 40, count)] + rng.normal(size=(count, args.dim)) * 0.55
        data = (data / np.linalg.norm(data, axis=1, keepdims=True)).astype(np.float32)
        probes = data[rng.choice(count, args.queries, replace=False)]

        exact = [set(np.argsort(-(data @ q))[: args.k].tolist()) for q in probes]
        chosen = cost.choose(count)
        entry = {"points": count, "cost_model_chose": chosen.value, "strategies": []}
        print(f"{count:,} points — cost model chooses {chosen.value}")

        for strategy in (Strategy.FLAT, Strategy.HNSW, Strategy.IVF_PQ):
            index = AdaptiveVectorIndex(args.dim, cost, f"bake{count}")
            t0 = time.perf_counter()
            for i, vector in enumerate(data):
                index.add(str(i), vector)
            append_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            try:
                index.force(strategy)
            except Exception as exc:
                print(f"  {strategy.value:<8} unavailable: {str(exc)[:70]}")
                entry["strategies"].append({"strategy": strategy.value,
                                            "failed": str(exc)[:120]})
                continue
            build_s = time.perf_counter() - t0

            latencies, hits = [], []
            for query, gold in zip(probes, exact):
                t0 = time.perf_counter()
                found = index.search(query, args.k)
                latencies.append((time.perf_counter() - t0) * 1000)
                hits.append(len(gold & {int(pid) for pid, _ in found}) / args.k)

            row = {"strategy": strategy.value, "append_s": round(append_s, 2),
                   "build_s": round(build_s, 2), "recall_at_k": round(float(np.mean(hits)), 4),
                   "latency_ms": percentiles(latencies)}
            entry["strategies"].append(row)
            print(f"  {strategy.value:<8} build {build_s:>7.2f}s   "
                  f"p50 {row['latency_ms']['p50']:>7.3f} ms   "
                  f"p95 {row['latency_ms']['p95']:>7.3f} ms   "
                  f"recall@{args.k} {row['recall_at_k']:.3f}")

        ran = [r for r in entry["strategies"] if "latency_ms" in r]
        if ran:
            fastest = min(ran, key=lambda r: r["latency_ms"]["p50"])
            entry["fastest"] = fastest["strategy"]
            entry["cost_model_correct"] = fastest["strategy"] == chosen.value
            verdict = ("correct" if entry["cost_model_correct"]
                       else f"WRONG — {fastest['strategy']} was faster")
            print(f"  -> fastest is {fastest['strategy']}; cost model {verdict}\n")
        report["corpora"].append(entry)

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
