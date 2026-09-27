"""Does the engine beat the interpreter? Run both and see.

    python3 scripts/qdrant_bakeoff.py --points 2000 --queries 64

Every stage of this node's hybrid retrieval — dense recall, sparse recall,
rank fusion, late-interaction rerank — is also a thing Qdrant does natively.
`QdrantStore.search_native()` expresses all four as one `query_points` call.
This script is the reason that call is allowed to become the default, or not:
it runs the same queries down both paths on the same corpus and reports
latency, how much of the top-k they agree on, and how much of the *ordering*
they agree on.

Two results are worth expecting rather than hoping for. On an embedded
deployment the engine is the client's pure-Python local mode, and this node's
index is a calibrated HNSW with a quantized tier, so the interpreter should
win — and if it does, this script says so and the default stays put. On a
Qdrant Server the engine is Rust with the postings already warm, and the
single call removes four round trips; there the trade flips.

The verdict line at the bottom is computed from the numbers, not typed in.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aegis.config import Settings                                # noqa: E402
from aegis.memory.filters import Condition, Filter, Op           # noqa: E402
from aegis.node import EdgeNode                                  # noqa: E402
from aegis.retrieval.fusion import reciprocal_rank_fusion        # noqa: E402

O, R, B, D, G, Y = ("\033[38;5;208m", "\033[0m", "\033[1m", "\033[2m",
                    "\033[38;5;42m", "\033[38;5;220m")

WORDS = ("conveyor bearing vibration raceway gantry pump seal night shift line torque "
         "spindle coolant hydraulic valve actuator encoder relay inverter motor gearbox "
         "alignment lubricant thermostat calibration overload phase rotor stator").split()


def synthetic(i: int, rng: random.Random) -> str:
    return f"observation {i}: " + " ".join(rng.sample(WORDS, 8))


def mean(values: list[float]) -> float:
    # Not `statistics.fmean`: this repo has its own scripts/statistics.py, and
    # a script's own directory precedes the standard library on sys.path.
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return ordered[index]


def rank_agreement(left: list[str], right: list[str]) -> float:
    """Average prefix overlap — the metric a top-k answer actually feels.

    Plain set overlap calls two orderings of the same five memories a perfect
    match. For an operator reading the first result, it is not.
    """
    depth = min(len(left), len(right))
    if depth == 0:
        return 0.0
    return sum(len(set(left[:i]) & set(right[:i])) / i for i in range(1, depth + 1)) / depth


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=2_000)
    parser.add_argument("--queries", type=int, default=64)
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--json", type=str, default="")
    args = parser.parse_args()

    node = EdgeNode(Settings())
    await node.start()
    store = node.store.store
    if not hasattr(store, "search_native"):
        print(f"{Y}this node is not on Qdrant — nothing to bake off against{R}")
        await node.stop()
        return 1

    quota = node.tenants.tenants["default"].quota
    quota.max_points = quota.max_ingest_per_minute = 10_000_000
    quota.max_qps, quota.max_bytes = 1e9, 1 << 40

    rng = random.Random(0)
    print(f"{O}{B}qdrant bake-off{R}  backend {B}{store.backend}{R}, "
          f"{args.points:,} points, {args.queries} queries, k={args.k}")
    print(f"{D}schema: {json.dumps(store.hybrid.snapshot()['schema'])}{R}\n")

    started = time.perf_counter()
    for i in range(args.points):
        await node.remember(synthetic(i, rng), "episodic")
    ingest = time.perf_counter() - started
    print(f"{D}ingested in {ingest:.1f}s ({args.points / max(ingest, 1e-9):.1f} docs/s) — "
          f"both paths were written by the same upsert{R}\n")

    queries = [" ".join(rng.sample(WORDS, 4)) for _ in range(args.queries)]
    filtered = Filter(must=[Condition("collection", Op.EQ, "episodic"),
                            Condition("confidence", Op.GTE, 0.0)])

    # Exact cosine top-k over the whole corpus, computed once per query by
    # brute force. Neither path is ground truth for the other: this is.
    ids = [pid for pid, point in node.store.points.items() if point.collection == "episodic"]
    matrix = np.asarray([node.store.points[pid].dense for pid in ids], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    unit = matrix / np.where(norms == 0, 1.0, norms)

    def exact(vector: np.ndarray, k: int) -> list[str]:
        query = np.asarray(vector, dtype=np.float32)
        query = query / (np.linalg.norm(query) or 1.0)
        scores = unit @ query
        return [ids[i] for i in np.argsort(-scores)[:k]]

    local_ms: list[float] = []
    local_dense_recall: list[float] = []
    native_dense_recall: list[float] = []
    dense_overlaps: list[float] = []
    sparse_overlaps: list[float] = []
    constant_overlaps: list[float] = []
    native_ms: list[float] = []
    overlaps: list[float] = []
    ranks: list[float] = []
    refusals = 0
    filtered_overlaps: list[float] = []

    for query in queries:
        vector = node.embedder.embed_sync([query])[0]
        sparse = node.sparse.encode(query)
        fetch = max(args.k * 6, 24)

        t0 = time.perf_counter()
        dense_hits = store.search_dense("episodic", vector, fetch)
        sparse_hits = store.search_sparse("episodic", sparse, fetch)
        fused = reciprocal_rank_fusion(dense_hits, sparse_hits)
        local = [hit.point_id for hit in fused][: args.k]
        local_ms.append((time.perf_counter() - t0) * 1000)

        # decompose the disagreement: dense recall, sparse recall, then fusion
        truth = set(exact(vector, args.k))
        local_dense = [pid for pid, _ in store.search_dense("episodic", vector, args.k)]
        engine_dense, dense_plan = store.search_native("episodic", vector, {}, args.k,
                                                       fetch=fetch, mode="dense")
        engine_dense_ids = [pid for pid, _ in engine_dense]
        if not dense_plan.fell_back:
            local_dense_recall.append(len(truth & set(local_dense)) / max(len(truth), 1))
            native_dense_recall.append(len(truth & set(engine_dense_ids)) / max(len(truth), 1))
            dense_overlaps.append(
                len(set(local_dense) & set(engine_dense_ids)) / max(len(local_dense), 1))
        local_sparse_ids = [pid for pid, _ in store.search_sparse("episodic", sparse, args.k)]
        engine_sparse, sparse_plan = store.search_native("episodic", vector, sparse, args.k,
                                                        fetch=fetch, mode="sparse")
        if not sparse_plan.fell_back and local_sparse_ids:
            sparse_overlaps.append(len(set(local_sparse_ids) & {pid for pid, _ in engine_sparse})
                                   / max(len(local_sparse_ids), 1))

        hits, plan = store.search_native("episodic", vector, sparse, args.k, fetch=fetch)
        if plan.fell_back:
            refusals += 1
            continue
        native = [pid for pid, _ in hits]
        # Attribute whatever gap is left to the fusion constant rather than
        # guessing at it: Qdrant's RRF is unweighted with a constant of 1,
        # ours weights dense above sparse with a constant of 60. Fusing the
        # interpreter's own candidates on the engine's terms isolates that.
        engine_terms = reciprocal_rank_fusion(dense_hits, sparse_hits, k=1,
                                              dense_weight=1.0, sparse_weight=1.0)
        constant_overlaps.append(
            len({h.point_id for h in engine_terms[: args.k]} & set(native))
            / max(len(native), 1))
        native_ms.append(plan.total_ms)
        overlaps.append(len(set(local) & set(native)) / max(len(local), 1))
        ranks.append(rank_agreement(local, native))

        # the same query with a filter the engine *can* express, to prove the
        # pre-filter path agrees too rather than only the unfiltered one
        f_hits, f_plan = store.search_native("episodic", vector, sparse, args.k,
                                            fetch=fetch, spec=filtered)
        if not f_plan.fell_back:
            filtered_overlaps.append(
                len(set(pid for pid, _ in f_hits) & set(native)) / max(len(native), 1))

    if not native_ms:
        print(f"{Y}the engine declined every query ({refusals} refusals){R}")
        await node.stop()
        return 1

    rows = [
        ("latency p50 (ms)", percentile(local_ms, 0.5), percentile(native_ms, 0.5)),
        ("latency p95 (ms)", percentile(local_ms, 0.95), percentile(native_ms, 0.95)),
        ("latency mean (ms)", mean(local_ms), mean(native_ms)),
    ]
    print(f"{B}{'':<20}{'interpreter':>14}{'engine':>12}{'ratio':>9}{R}")
    for name, left, right in rows:
        ratio = (left / right) if right else float("inf")
        colour = G if ratio >= 1.0 else Y
        print(f"{name:<20}{left:>14.3f}{right:>12.3f}{colour}{ratio:>8.2f}×{R}")

    p50_local, p50_native = percentile(local_ms, 0.5), percentile(native_ms, 0.5)
    overlap = mean(overlaps)
    rank = mean(ranks)
    print()
    print(f"{'top-k overlap':<20}{overlap:>14.3f}{D}   how much of the same set{R}")
    print(f"{'rank agreement':<20}{rank:>14.3f}{D}   how much of the same order{R}")
    if filtered_overlaps:
        print(f"{'pre-filter overlap':<20}{mean(filtered_overlaps):>14.3f}"
              f"{D}   engine-side filter vs engine-side unfiltered{R}")
    print(f"\n{B}where the two paths part company{R}")
    print(f"{'dense-only overlap':<20}{mean(dense_overlaps):>14.3f}"
          f"{D}   same query, dense space only{R}")
    print(f"{'sparse-only overlap':<20}{mean(sparse_overlaps):>14.3f}"
          f"{D}   same query, sparse space only{R}")
    print(f"{'same-constant overlap':<20}{mean(constant_overlaps):>14.3f}"
          f"{D}   interpreter fusion re-run on the engine's RRF constant{R}")
    print(f"{'dense recall@k':<20}{mean(local_dense_recall):>14.3f}"
          f"{mean(native_dense_recall):>12.3f}{D}   against exact brute force{R}")
    print()
    print(f"{'engine refusals':<20}{refusals:>14d}{D}   queries the engine declined; "
          f"those stay on the interpreter{R}")
    print(f"{'round trips':<20}{'4 → 1':>14}{D}   recall, recall, fuse, rerank in one call{R}")

    faster = p50_native < p50_local
    # "Agrees" is judged per stage, not on the top-k alone. Two paths can
    # compute identical dense recall, identical sparse recall and identical
    # fusion *arithmetic*, and still order the final five differently because
    # one of them weights the spaces — which is a parameter, not a defect. So
    # the verdict names which of the two it is instead of averaging them into
    # a number nobody can act on.
    same_pipeline = (mean(dense_overlaps) >= 0.99 and mean(sparse_overlaps) >= 0.99
                     and mean(constant_overlaps) >= 0.99)
    if same_pipeline and faster:
        verdict = "engine computes the same pipeline and is faster: make it the default"
    elif same_pipeline:
        verdict = ("engine computes the same pipeline but is slower here: keep it "
                   "available, default unchanged")
    elif overlap >= 0.8 and rank >= 0.7:
        verdict = "engine agrees within tolerance but the difference is unattributed"
    else:
        verdict = "engine disagrees for a reason this run did not attribute: do not promote it"
    agrees = same_pipeline
    colour = G if same_pipeline else Y
    print(f"\n{colour}{B}verdict{R} {colour}{verdict}{R}")
    if same_pipeline and overlap < 1.0:
        print(f"{D}the top-k differ on {1 - overlap:.0%} of slots and all of it is the fusion "
              f"constant: Qdrant's RRF is unweighted with a constant of 1, this node's "
              f"weights dense above sparse with a constant of 60. Re-run the interpreter's "
              f"fusion on the engine's constant and the two agree exactly "
              f"({mean(constant_overlaps):.3f}).{R}")
    print(f"{D}derived, not typed: faster={faster}, same_pipeline={same_pipeline} "
          f"(per-stage overlap ≥ 0.99 in both spaces and under a shared fusion constant){R}")

    # 4. and what the router does with all of this, which is the point: the
    #    numbers above are only worth measuring if something acts on them.
    router = node.pipeline.router
    before = router.snapshot()
    for query in queries[:24]:
        await node.pipeline.search(query, k=args.k, collection="episodic")
    after = router.snapshot()
    routed_engine = after["to_engine"] - before["to_engine"]
    routed_index = after["to_index"] - before["to_index"]
    print(f"\n{B}what the router did with 24 live queries{R}")
    print(f"{'to the engine':<20}{routed_engine:>14d}")
    print(f"{'to the local index':<20}{routed_index:>14d}")
    print(f"{'recall budget':<20}{after['recall_budget_ms']:>14.1f}{D}   ms, "
          f"{int(router.SHARE_OF_OBJECTIVE * 100)}% of the query objective{R}")
    print(f"{'engine p95 seen':<20}{after['engine_p95_ms']:>14.1f}{D}   ms{R}")
    print(f"{D}last decision: {after['last']['reason']}{R}")

    payload = {
        "backend": store.backend, "points": args.points, "queries": len(native_ms),
        "k": args.k, "refusals": refusals,
        "local_ms": {"p50": round(p50_local, 3), "p95": round(percentile(local_ms, 0.95), 3)},
        "native_ms": {"p50": round(p50_native, 3), "p95": round(percentile(native_ms, 0.95), 3)},
        "overlap": round(overlap, 3), "rank_agreement": round(rank, 3),
        "dense_overlap": round(mean(dense_overlaps), 3),
        "sparse_overlap": round(mean(sparse_overlaps), 3),
        "same_constant_overlap": round(mean(constant_overlaps), 3),
        "dense_recall_at_k": {"interpreter": round(mean(local_dense_recall), 3),
                              "engine": round(mean(native_dense_recall), 3)},
        "pre_filter_overlap": (round(mean(filtered_overlaps), 3)
                               if filtered_overlaps else None),
        "faster": faster, "same_pipeline": same_pipeline, "verdict": verdict,
        "hybrid": store.hybrid.snapshot(),
        "router": {"to_engine": routed_engine, "to_index": routed_index, **after},
    }
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"{D}wrote {args.json}{R}")
    await node.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
