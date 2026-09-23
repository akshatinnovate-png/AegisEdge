"""Conformal retrieval: calibrated confidence and principled abstention.

A similarity score is not a probability. 0.83 means nothing on its own — it
depends on the encoder, the corpus, the query distribution and the quantizer
the thermal governor happened to swap in ten minutes ago. Systems that
threshold raw scores are guessing, and on an edge device a confident wrong
answer is worse than no answer: nobody is watching to catch it.

Split conformal prediction gives a distribution-free guarantee instead. From a
calibration set of queries whose correct answer is known (the node collects
these from feedback), take the nonconformity scores of the true answers, and
the (1-alpha) empirical quantile becomes a threshold with a finite-sample
coverage guarantee: the returned set contains the correct answer at least
(1-alpha) of the time, whatever the score distribution looks like.

Two consequences the product cares about:

* the node can **abstain** — "nothing here clears the bar" — with a stated
  error rate rather than a hunch;
* the set size becomes a *measured* signal of ambiguity: a query needing
  eleven candidates to reach 90% coverage is a genuinely ambiguous query, and
  saying so is more useful than returning one confident-looking row.

Coverage is re-checked continuously against realised outcomes, so a drifting
encoder shows up as drifting coverage instead of silent degradation.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CalibrationSample:
    score: float                    # score of the *correct* answer
    top_score: float                # best score in that query's candidate set
    at: float = field(default_factory=time.time)


@dataclass
class ConformalResult:
    prediction_set: list[str]
    threshold: float
    alpha: float
    coverage_target: float
    abstained: bool
    ambiguity: float
    calibrated_on: int
    guarantee: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "prediction_set": self.prediction_set, "set_size": len(self.prediction_set),
            "threshold": round(self.threshold, 5), "alpha": self.alpha,
            "coverage_target": self.coverage_target, "abstained": self.abstained,
            "ambiguity": round(self.ambiguity, 4), "calibrated_on": self.calibrated_on,
            "guarantee": self.guarantee,
        }


class ConformalPredictor:
    MIN_CALIBRATION = 20

    def __init__(self, alpha: float = 0.1, window: int = 512) -> None:
        self.alpha = alpha
        self.calibration: deque[CalibrationSample] = deque(maxlen=window)
        self.realised: deque[bool] = deque(maxlen=window)     # was the truth in the set?
        self.predictions = 0
        self.abstentions = 0
        self.set_sizes: deque[int] = deque(maxlen=window)

    # -- calibration ------------------------------------------------------

    def observe(self, correct_score: float, top_score: float) -> None:
        """One labelled outcome: what the correct answer scored, and the best score."""
        self.calibration.append(CalibrationSample(float(correct_score), float(top_score)))

    @staticmethod
    def _nonconformity(sample: CalibrationSample) -> float:
        """How badly the correct answer trailed the best candidate.

        Using the *gap* rather than the raw score makes calibration robust to
        the encoder or quantizer changing the absolute scale underneath us —
        which the thermal governor does on purpose.
        """
        return max(0.0, sample.top_score - sample.score)

    @property
    def calibrated(self) -> bool:
        return len(self.calibration) >= self.MIN_CALIBRATION

    def quantile(self) -> float:
        """The finite-sample corrected (1-alpha) quantile of nonconformity."""
        if not self.calibration:
            return float("inf")
        scores = sorted(self._nonconformity(s) for s in self.calibration)
        n = len(scores)
        # ceil((n+1)(1-alpha)) / n  — the split-conformal correction; without it
        # coverage is only asymptotic, and an edge node's calibration set is small
        rank = math.ceil((n + 1) * (1.0 - self.alpha))
        if rank > n:
            return float("inf")                       # too few samples to promise anything
        return scores[rank - 1]

    # -- prediction -------------------------------------------------------

    def predict(self, scored: list[tuple[str, float]], max_set: int = 10) -> ConformalResult:
        """Return every candidate within the calibrated gap of the best one."""
        self.predictions += 1
        if not scored:
            self.abstentions += 1
            return ConformalResult([], 0.0, self.alpha, 1 - self.alpha, True, 1.0,
                                   len(self.calibration), "no candidates")

        ordered = sorted(scored, key=lambda row: -row[1])
        top = ordered[0][1]

        if not self.calibrated:
            # Honest fallback: no guarantee is claimed until there is evidence.
            self.set_sizes.append(min(len(ordered), max_set))
            return ConformalResult(
                [pid for pid, _ in ordered[:max_set]], float("inf"), self.alpha,
                1 - self.alpha, False, 1.0, len(self.calibration),
                f"uncalibrated — needs {self.MIN_CALIBRATION - len(self.calibration)} "
                f"more labelled outcomes before coverage can be claimed",
            )

        gap = self.quantile()
        prediction_set = [pid for pid, score in ordered if (top - score) <= gap][:max_set]
        # An empty or corpus-wide set both mean "this query is not answerable
        # at the requested confidence" — abstain rather than pretend.
        abstain = not prediction_set or len(prediction_set) >= min(max_set, len(ordered))
        ambiguity = len(prediction_set) / max(len(ordered), 1)
        if abstain:
            self.abstentions += 1
        self.set_sizes.append(len(prediction_set))
        return ConformalResult(
            prediction_set, gap, self.alpha, 1 - self.alpha, abstain, ambiguity,
            len(self.calibration),
            f"contains the correct answer with probability >= {1 - self.alpha:.0%} "
            f"(split conformal, n={len(self.calibration)})",
        )

    def record_outcome(self, covered: bool) -> None:
        """Feed a realised outcome back so coverage can be audited, not assumed."""
        self.realised.append(bool(covered))

    @property
    def empirical_coverage(self) -> float | None:
        if len(self.realised) < 10:
            return None
        return sum(self.realised) / len(self.realised)

    def drift(self) -> dict[str, Any]:
        """Is realised coverage still meeting the promise?"""
        coverage = self.empirical_coverage
        if coverage is None:
            return {"status": "insufficient_evidence", "samples": len(self.realised)}
        target = 1 - self.alpha
        # binomial standard error on the observed coverage
        error = math.sqrt(max(target * (1 - target), 1e-9) / len(self.realised))
        breach = coverage < target - 2 * error
        return {
            "status": "breached" if breach else "holding",
            "empirical_coverage": round(coverage, 4), "target": target,
            "tolerance": round(2 * error, 4), "samples": len(self.realised),
            "detail": ("coverage has fallen below the guarantee — recalibrate, the encoder "
                       "or the corpus has drifted" if breach else "coverage is meeting the guarantee"),
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha, "coverage_target": 1 - self.alpha,
            "calibrated": self.calibrated, "calibration_samples": len(self.calibration),
            "threshold": None if not self.calibrated else round(self.quantile(), 5),
            "predictions": self.predictions, "abstentions": self.abstentions,
            "abstention_rate": round(self.abstentions / self.predictions, 4) if self.predictions else 0.0,
            "avg_set_size": round(float(np.mean(self.set_sizes)), 2) if self.set_sizes else 0.0,
            "drift": self.drift(),
        }
