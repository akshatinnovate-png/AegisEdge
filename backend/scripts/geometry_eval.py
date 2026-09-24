"""Does corpus-adapted geometry actually retrieve better? Measure, do not assume.

Whitening a pretrained embedding space is a well-published idea that is also
easy to get wrong: the transform divides by the square root of each
eigenvalue, so in a space whose effective rank is a fraction of its dimension
it will happily amplify two hundred directions of rounding error. The only way
to know which side of that line a configuration falls on is an evaluation with
ground truth.

The task here is paraphrase retrieval, built from the corpus itself so no
labels have to be invented: take a document, drop a random share of its words,
and ask the index to find the document it came from. The correct answer is
known by construction, it exercises exactly the property retrieval needs
(robustness to surface form), and nothing about it can be tuned to flatter the
transform.

Reported for each configuration: recall@1, MRR@10, and the geometry
diagnostics that explain the result.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aegis.inference.adaptation import paired_bootstrap                # noqa: E402
from aegis.inference.geometry import AnisotropyProbe, CorpusGeometry   # noqa: E402
from aegis.inference.onnx_runtime import open_sessions                 # noqa: E402
from aegis.memory.quantize import BinaryQuantizer                      # noqa: E402
from aegis.memory.rabitq import RaBitQ, adaptive_shortlist             # noqa: E402

SUBJECTS = ["bay 3 conveyor", "spindle SP-9920", "line 2 gantry", "coolant loop",
            "bearing housing BX-7741", "the interlock", "servo drive 4",
            "hydraulic pump", "tool changer", "chiller unit", "vibration sensor",
            "torque transducer", "encoder ring", "belt tensioner", "gearbox",
            "the packing head", "conveyor motor M12", "pressure relief valve"]
VERBS = ["vibration crossed", "pressure dropped to", "temperature climbed to",
         "current drew", "torque peaked at", "runout measured", "clearance fell to",
         "speed held at", "backlash grew to", "flow rate settled at"]
TAILS = ["during the night shift", "after the tool change", "before the interlock fired",
         "on the second pass", "while the line was in manual", "at the start of the run",
         "after maintenance replaced the housing", "during the ramp to full rate",
         "immediately after the restart", "once the chiller caught up"]


def corpus(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    values = [f"{v / 10:.1f} {u}" for v, u in
              itertools.product(range(5, 95, 3), ["mm/s", "bar", "C", "A", "nm", "um", "rpm"])]
    return [f"{rng.choice(SUBJECTS)} {rng.choice(VERBS)} {rng.choice(values)} "
            f"{rng.choice(TAILS)}" for _ in range(n)]


def paraphrase(text: str, keep: float, rng: random.Random) -> str:
    words = text.split()
    chosen = [w for w in words if rng.random() < keep]
    return " ".join(chosen) if len(chosen) >= 2 else " ".join(words[:3])


def reciprocal_ranks(doc_vectors: np.ndarray, query_vectors: np.ndarray,
                     truth: np.ndarray, k: int = 10) -> np.ndarray:
    order = np.argsort(-(query_vectors @ doc_vectors.T), axis=1)[:, :k]
    scores = np.zeros(order.shape[0], dtype=np.float64)
    for index, (row, gold) in enumerate(zip(order, truth)):
        found = np.flatnonzero(row == gold)
        if found.size:
            scores[index] = 1.0 / (found[0] + 1)
    return scores


def evaluate(doc_vectors: np.ndarray, query_vectors: np.ndarray,
             truth: np.ndarray, k: int = 10) -> dict[str, float]:
    scores = reciprocal_ranks(doc_vectors, query_vectors, truth, k)
    return {"recall_at_1": round(float(np.mean(scores == 1.0)), 4),
            "mrr_at_10": round(float(scores.mean()), 4)}


def quantization(data: np.ndarray, queries: np.ndarray, k: int = 10) -> dict:
    """Cold-tier recall under 1-bit codes, RaBitQ against plain sign bits."""
    dim = data.shape[1]
    rabit = RaBitQ(dim)
    rabit.fit(data)
    codes = rabit.encode(data)
    sign_codes = BinaryQuantizer.encode(data)
    out: dict[str, float] = {}
    for multiplier in (1, 2, 6):
        rabit_hits, sign_hits = [], []
        for query in queries:
            exact = data @ query
            gold = set(np.argsort(-exact)[:k].tolist())
            estimate, _ = rabit.estimate(rabit.prepare(query), codes)
            rabit_hits.append(len(gold & set(np.argsort(-estimate)[:k * multiplier].tolist())) / k)
            approx = BinaryQuantizer.similarity(
                np.packbits(query > 0).reshape(1, -1), sign_codes, dim)
            sign_hits.append(len(gold & set(np.argsort(-approx)[:k * multiplier].tolist())) / k)
        out[f"rabitq_recall_top{multiplier}k"] = round(float(np.mean(rabit_hits)), 4)
        out[f"signbit_recall_top{multiplier}k"] = round(float(np.mean(sign_hits)), 4)
    shortlists = []
    for query in queries[:40]:
        estimate, bound = rabit.estimate(rabit.prepare(query), codes)
        shortlists.append(len(adaptive_shortlist(estimate, bound, k)))
    out["adaptive_shortlist_mean"] = round(float(np.mean(shortlists)), 1)
    out["adaptive_shortlist_share"] = round(float(np.mean(shortlists)) / len(data), 4)
    out["bytes_per_vector"] = codes.nbytes // max(len(codes), 1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=int, default=8000)
    parser.add_argument("--queries", type=int, default=600)
    parser.add_argument("--keep", type=float, default=0.55)
    parser.add_argument("--models", default="models")
    parser.add_argument("--out", default="../testlogs/geometry-eval.json")
    args = parser.parse_args()

    embedder, _, bundle = open_sessions(Path(args.models), 128)
    rng = random.Random(11)
    documents = corpus(args.docs, seed=5)
    picked = rng.sample(range(len(documents)), args.queries)
    queries = [paraphrase(documents[i], args.keep, rng) for i in picked]
    truth = np.asarray(picked)

    def encode(texts: list[str]) -> np.ndarray:
        return np.vstack([embedder.encode(texts[i:i + 128])
                          for i in range(0, len(texts), 128)]).astype(np.float32)

    raw_docs, raw_queries = encode(documents), encode(queries)
    report: dict = {"model": bundle.source, "dim": int(embedder.dim),
                    "documents": args.docs, "queries": args.queries,
                    "paraphrase_keep": args.keep, "configurations": []}

    baseline_ranks = reciprocal_ranks(raw_docs, raw_queries, truth)
    baseline = {"name": "raw (as shipped)",
                "geometry": AnisotropyProbe.measure(raw_docs),
                "retrieval": evaluate(raw_docs, raw_queries, truth),
                "quantization": quantization(raw_docs, raw_queries[:60])}
    report["configurations"].append(baseline)
    print(f"raw (as shipped)                 "
          f"R@1 {baseline['retrieval']['recall_at_1']:.4f}  "
          f"MRR {baseline['retrieval']['mrr_at_10']:.4f}  "
          f"pair-cos {baseline['geometry']['mean_random_pair_cosine']:+.4f}  "
          f"eff.rank {baseline['geometry']['effective_rank']:.1f}", flush=True)

    grid = [(0, 0.95), (1, 0.95), (2, 0.95), (0, 0.99), (1, 0.99), (0, 0.90), (1, 0.80)]
    for drop, target in grid:
        geometry = CorpusGeometry(embedder.dim, drop_components=drop, energy_target=target)
        geometry.observe(raw_docs)
        version = geometry.fit()
        if version is None:
            continue
        docs = geometry.transform(raw_docs)
        qs = geometry.transform(raw_queries)
        interval = paired_bootstrap(baseline_ranks,
                                    reciprocal_ranks(docs, qs, truth))
        entry = {"name": f"whitened drop={drop} energy={target}",
                 "version": version.as_dict(), "rank": geometry.rank,
                 "geometry": AnisotropyProbe.measure(docs),
                 "retrieval": evaluate(docs, qs, truth),
                 "significance": {"delta": round(interval["mean"], 4),
                                  "ci95": [round(interval["low"], 4),
                                           round(interval["high"], 4)],
                                  "significant": bool(interval["low"] > 0)},
                 "quantization": quantization(docs, qs[:60]),
                 "nested_ladder": geometry.nested_ladder()}
        report["configurations"].append(entry)
        print(f"whiten drop={drop} energy={target:<4}  rank {geometry.rank:3d}  "
              f"R@1 {entry['retrieval']['recall_at_1']:.4f}  "
              f"MRR {entry['retrieval']['mrr_at_10']:.4f}  "
              f"delta {interval['mean']:+.4f} CI95 "
              f"[{interval['low']:+.4f},{interval['high']:+.4f}]"
              f"{'  SIGNIFICANT' if interval['low'] > 0 else ''}", flush=True)

    # Rank by the lower confidence bound, not the point estimate: an effect
    # nobody can distinguish from zero is not evidence, however large it looks.
    significant = [c for c in report["configurations"]
                   if c.get("significance", {}).get("significant")]
    report["significant_configurations"] = [c["name"] for c in significant]
    if significant:
        best = max(significant, key=lambda c: c["significance"]["ci95"][0])
        lift = best["significance"]["delta"]
        report["best"] = best["name"]
        report["mrr_lift_over_raw"] = lift
        report["verdict"] = (f"adopt: {best['name']} (+{lift:.4f} MRR, 95% CI "
                             f"{best['significance']['ci95']})")
    else:
        report["best"] = "raw (as shipped)"
        report["mrr_lift_over_raw"] = 0.0
        report["verdict"] = ("keep raw: no configuration beat the shipped space by a "
                             "margin distinguishable from sampling noise")
    print("\n" + report["verdict"])

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
