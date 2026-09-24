"""The node decides its own representation, by measurement, with no labels.

Every adaptation in this package is a plausible idea with a published result
behind it, and plausible is not the same as true *here*. Two of them were
measured on this encoder against real prose:

    masked mean (as shipped)        MRR@10  0.7827      baseline
    + corpus whitening              MRR@10  0.7989      +0.0162   adopt
    + SIF pooling (a=1e-3)          MRR@10  0.7457      -0.0370   reject
    + SIF pooling (a=1e-4)          MRR@10  0.6376      -0.1451   reject

Smooth Inverse Frequency is not a bad idea — it is one of the most cited
results in sentence embedding. It is simply the wrong idea for *this* model,
whose token table was distilled specifically for mean pooling and has
therefore already absorbed the frequency correction SIF exists to apply.
Shipping it on the strength of the citation would have cost 5% of retrieval
quality, silently, on every device.

So the node does not take anybody's word for it, including its own author's.
This gate builds an evaluation out of the corpus the device has actually
ingested, scores each candidate configuration on it, and arms exactly the ones
that win by a margin. A candidate that cannot prove itself stays off, and the
refusal is recorded with the numbers that caused it.

The evaluation needs no labels because the ground truth is constructed: take a
stored document, delete half its words at random, and the document it came
from is the correct answer by definition. It measures the property retrieval
actually needs — robustness to surface form — and nothing about it can be
tuned to flatter a candidate.
"""
from __future__ import annotations

import random
import time
from typing import Any, Callable, Iterable

import numpy as np


class ProbeSet:
    """A paraphrase-retrieval task built from the node's own memories."""

    __slots__ = ("documents", "queries", "truth", "built_at")

    def __init__(self, documents: list[str], queries: list[str],
                 truth: np.ndarray) -> None:
        self.documents = documents
        self.queries = queries
        self.truth = truth
        self.built_at = time.time()

    def __len__(self) -> int:
        return len(self.queries)

    @classmethod
    def build(cls, texts: list[str], queries: int = 200, keep: float = 0.5,
              seed: int = 17) -> "ProbeSet | None":
        usable = [t for t in texts if len(t.split()) >= 6]
        if len(usable) < max(64, queries // 2):
            return None
        rng = random.Random(seed)
        picked = rng.sample(range(len(usable)), min(queries, len(usable)))
        probes = []
        for index in picked:
            words = usable[index].split()
            kept = [w for w in words if rng.random() < keep]
            probes.append(" ".join(kept) if len(kept) >= 3 else " ".join(words[:4]))
        return cls(usable, probes, np.asarray(picked))

    def reciprocal_ranks(self, document_vectors: np.ndarray,
                         query_vectors: np.ndarray, k: int = 10) -> np.ndarray:
        """Per-query reciprocal rank. Kept per query, not averaged away.

        The mean is what gets reported, but the vector is what lets the gate
        put a confidence interval on a difference between two configurations —
        and without one it cannot tell a real gain from a lucky sample.
        """
        similarity = query_vectors @ document_vectors.T
        ranking = np.argsort(-similarity, axis=1)[:, :k]
        scores = np.zeros(ranking.shape[0], dtype=np.float64)
        for index, (row, gold) in enumerate(zip(ranking, self.truth)):
            hit = np.flatnonzero(row == gold)
            if hit.size:
                scores[index] = 1.0 / (hit[0] + 1)
        return scores

    def score(self, document_vectors: np.ndarray, query_vectors: np.ndarray,
              k: int = 10) -> dict[str, float]:
        """MRR@k and recall@1 — the query must find the document it came from."""
        scores = self.reciprocal_ranks(document_vectors, query_vectors, k)
        return {"mrr_at_10": float(scores.mean()),
                "recall_at_1": float(np.mean(scores == 1.0))}


def paired_bootstrap(baseline: np.ndarray, candidate: np.ndarray,
                     resamples: int = 2000, seed: int = 4,
                     alpha: float = 0.05) -> dict[str, float]:
    """Confidence interval on the per-query difference between two configurations.

    Retrieval metrics on a few hundred probes are noisy enough that a real
    effect and a lucky sample look identical in the mean. This one is not
    hypothetical: whitening measured +0.0175 MRR on one probe sample of this
    corpus and -0.0001 on another drawn from the same text. A gate that
    adopted on the first and rejected on the second is not measuring the
    transform, it is measuring which documents it happened to pick.

    The queries are paired — the same probe scored under both configurations —
    so the difference is taken per query and resampled, which removes the
    variance from "some probes are simply harder" and leaves only the variance
    of the effect itself.
    """
    difference = np.asarray(candidate, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    n = difference.size
    if n < 8:
        return {"mean": float(difference.mean()) if n else 0.0,
                "low": float("-inf"), "high": float("inf"), "resamples": 0}
    rng = np.random.default_rng(seed)
    means = difference[rng.integers(0, n, size=(resamples, n))].mean(axis=1)
    return {"mean": float(difference.mean()),
            "low": float(np.quantile(means, alpha / 2)),
            "high": float(np.quantile(means, 1 - alpha / 2)),
            "resamples": int(resamples),
            "wins": int((difference > 0).sum()), "losses": int((difference < 0).sum()),
            "ties": int((difference == 0).sum())}


class Decision:
    """What was tried, what it scored, and whether it was armed."""

    __slots__ = ("candidate", "metric", "baseline", "delta", "adopted", "reason",
                 "at", "probes", "detail")

    def __init__(self, candidate: str, metric: float, baseline: float, adopted: bool,
                 reason: str, probes: int, detail: dict[str, Any] | None = None) -> None:
        self.candidate = candidate
        self.metric = metric
        self.baseline = baseline
        self.delta = metric - baseline
        self.adopted = adopted
        self.reason = reason
        self.at = time.time()
        self.probes = probes
        self.detail = detail or {}

    def as_dict(self) -> dict[str, Any]:
        return {"candidate": self.candidate, "mrr_at_10": round(self.metric, 4),
                "baseline": round(self.baseline, 4), "delta": round(self.delta, 4),
                "adopted": self.adopted, "reason": self.reason, "at": self.at,
                "probes": self.probes, **self.detail}


class AdaptationGate:
    """Score candidate embedding configurations; arm only what wins.

    Deliberately conservative. An adaptation changes what every stored vector
    *means*, so adopting one is a migration, and a migration justified by noise
    is worse than no migration at all. Hence: a minimum corpus, a minimum probe
    count, and a margin large enough that a rerun on different samples would
    not flip the sign.
    """

    def __init__(self, min_documents: int = 500, min_probes: int = 100,
                 margin: float = 0.005, history: int = 32) -> None:
        self.min_documents = int(min_documents)
        self.min_probes = int(min_probes)
        self.margin = float(margin)
        self.resamples = 2000
        self.history: list[Decision] = []
        self._history_cap = history
        self.evaluations = 0
        self.last_run: float | None = None
        self.adopted: str | None = None

    def _record(self, decision: Decision) -> Decision:
        self.history.append(decision)
        if len(self.history) > self._history_cap:
            del self.history[: len(self.history) - self._history_cap]
        return decision

    def evaluate(self, texts: list[str],
                 encode: Callable[[list[str]], np.ndarray],
                 candidates: Iterable[tuple[str, Callable[[np.ndarray], np.ndarray] | None]],
                 queries: int = 200) -> dict[str, Any]:
        """Run the probe task over the baseline and each candidate transform.

        `encode` must produce the *baseline* embedding for a list of texts; a
        candidate is a post-transform applied to those vectors. Candidates that
        change tokenisation or pooling instead supply their own encoder by
        closing over it, which costs one extra pass over the probe set and
        nothing else.
        """
        if len(texts) < self.min_documents:
            return {"ran": False,
                    "reason": f"corpus too small: {len(texts)} < {self.min_documents}"}
        probes = ProbeSet.build(texts, queries=queries)
        if probes is None or len(probes) < self.min_probes:
            return {"ran": False, "reason": "not enough usable documents for a probe set"}

        self.evaluations += 1
        self.last_run = time.time()
        document_vectors = encode(probes.documents)
        query_vectors = encode(probes.queries)
        baseline_ranks = probes.reciprocal_ranks(document_vectors, query_vectors)
        baseline = {"mrr_at_10": float(baseline_ranks.mean()),
                    "recall_at_1": float(np.mean(baseline_ranks == 1.0))}

        results = [{"candidate": "baseline (as shipped)",
                    **{k: round(v, 4) for k, v in baseline.items()}}]
        best_name, best_score, best_detail = None, baseline["mrr_at_10"], {}
        best_interval: dict[str, float] = {}
        for name, transform in candidates:
            if transform is None:
                continue
            try:
                ranks = probes.reciprocal_ranks(transform(document_vectors),
                                                transform(query_vectors))
            except Exception as exc:                      # a candidate may not fit
                results.append({"candidate": name, "failed": str(exc)[:120]})
                continue
            interval = paired_bootstrap(baseline_ranks, ranks, self.resamples)
            scored = {"mrr_at_10": float(ranks.mean()),
                      "recall_at_1": float(np.mean(ranks == 1.0))}
            results.append({
                "candidate": name, **{k: round(v, 4) for k, v in scored.items()},
                "delta": round(interval["mean"], 4),
                "ci95": [round(interval["low"], 4), round(interval["high"], 4)],
                "significant": bool(interval["low"] > 0.0),
                "wins_losses": [interval.get("wins", 0), interval.get("losses", 0)]})
            # Rank candidates by the *lower* bound, not the point estimate: a
            # large effect nobody can distinguish from zero is not a better bet
            # than a small one that is solid.
            if interval["low"] > best_interval.get("low", 0.0) and interval["low"] > 0.0:
                best_name, best_score = name, scored["mrr_at_10"]
                best_detail = {k: round(v, 4) for k, v in scored.items()}
                best_interval = interval

        if best_name is None:
            decision = self._record(Decision(
                "none", baseline["mrr_at_10"], baseline["mrr_at_10"], False,
                "no candidate beat the shipped space with the difference "
                "distinguishable from sampling noise",
                len(probes), {}))
        elif best_interval["mean"] < self.margin:
            decision = self._record(Decision(
                best_name, best_score, baseline["mrr_at_10"], False,
                f"gain {best_interval['mean']:+.4f} is real (95% CI "
                f"[{best_interval['low']:+.4f}, {best_interval['high']:+.4f}]) but below "
                f"the {self.margin} margin that justifies re-embedding the corpus",
                len(probes), best_detail))
        else:
            decision = self._record(Decision(
                best_name, best_score, baseline["mrr_at_10"], True,
                f"gain {best_interval['mean']:+.4f}, 95% CI "
                f"[{best_interval['low']:+.4f}, {best_interval['high']:+.4f}] over "
                f"{len(probes)} paired paraphrase probes",
                len(probes), {**best_detail,
                              "ci95": [round(best_interval["low"], 4),
                                       round(best_interval["high"], 4)]}))
            self.adopted = best_name

        return {"ran": True, "probes": len(probes), "documents": len(probes.documents),
                "baseline": {k: round(v, 4) for k, v in baseline.items()},
                "results": results, "decision": decision.as_dict()}

    def snapshot(self) -> dict[str, Any]:
        return {
            "evaluations": self.evaluations, "last_run": self.last_run,
            "adopted": self.adopted, "margin": self.margin,
            "min_documents": self.min_documents, "min_probes": self.min_probes,
            "history": [d.as_dict() for d in self.history[-8:]],
            "resamples": self.resamples,
            "policy": ("an adaptation is armed only if it beats the shipped space on a "
                       "paraphrase task built from this device's own memories, by a "
                       "margin whose 95% paired-bootstrap interval excludes zero"),
        }
