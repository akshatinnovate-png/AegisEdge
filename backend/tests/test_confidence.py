"""Conformal calibration and diversity selection."""
from __future__ import annotations

import numpy as np
import pytest

from aegis.retrieval.conformal import ConformalPredictor
from aegis.retrieval.diversity import MaximalMarginalRelevance


def _calibrate(predictor: ConformalPredictor, rng, n: int = 200, spread: float = 0.08) -> None:
    for _ in range(n):
        top = float(rng.uniform(0.6, 1.0))
        predictor.observe(top - abs(float(rng.normal(0, spread))), top)


def test_uncalibrated_predictor_claims_no_guarantee():
    predictor = ConformalPredictor()
    result = predictor.predict([("a", 0.9), ("b", 0.4)])
    assert "uncalibrated" in result.guarantee
    assert result.threshold == float("inf")


def test_empirical_coverage_meets_the_target():
    rng = np.random.default_rng(0)
    predictor = ConformalPredictor(alpha=0.1)
    _calibrate(predictor, rng)

    covered = 0
    trials = 400
    for _ in range(trials):
        top = float(rng.uniform(0.6, 1.0))
        candidates = [("truth", top - abs(float(rng.normal(0, 0.08))))]
        candidates += [(f"d{i}", top - abs(float(rng.normal(0, 0.2)))) for i in range(9)]
        result = predictor.predict(candidates, max_set=10)
        hit = "truth" in result.prediction_set
        predictor.record_outcome(hit)
        covered += hit
    assert covered / trials >= 0.88               # target 0.90, finite-sample slack


def test_tighter_alpha_produces_larger_sets():
    rng = np.random.default_rng(1)
    loose = ConformalPredictor(alpha=0.2)
    strict = ConformalPredictor(alpha=0.01)
    _calibrate(loose, rng)
    _calibrate(strict, np.random.default_rng(1))
    assert strict.quantile() >= loose.quantile()


def test_drift_is_detected_when_coverage_falls():
    rng = np.random.default_rng(2)
    predictor = ConformalPredictor(alpha=0.1)
    _calibrate(predictor, rng, spread=0.02)       # calibrated on an easy distribution
    for _ in range(100):
        predictor.record_outcome(False)          # reality disagrees
    drift = predictor.drift()
    assert drift["status"] == "breached"
    assert "recalibrate" in drift["detail"]


def test_abstention_on_an_empty_candidate_set():
    predictor = ConformalPredictor()
    result = predictor.predict([])
    assert result.abstained and result.prediction_set == []


def test_mmr_displaces_near_duplicates():
    rng = np.random.default_rng(3)
    base = rng.normal(size=32).astype(np.float32)
    base /= np.linalg.norm(base)
    candidates = []
    for i in range(5):
        vector = base + 0.02 * rng.normal(size=32).astype(np.float32)
        vector /= np.linalg.norm(vector)
        candidates.append((f"dup{i}", 0.9 - 0.01 * i, vector))
    for i in range(3):
        vector = rng.normal(size=32).astype(np.float32)
        vector /= np.linalg.norm(vector)
        candidates.append((f"distinct{i}", 0.7 - 0.05 * i, vector))

    selected, report = MaximalMarginalRelevance().select(base, candidates, k=4)
    chosen = {pid for pid, _ in selected}
    assert sum(1 for pid in chosen if pid.startswith("dup")) <= 2
    assert report.redundancy_after < report.redundancy_before


def test_mmr_leaves_an_already_diverse_set_alone():
    rng = np.random.default_rng(4)
    candidates = []
    for i in range(6):
        vector = rng.normal(size=32).astype(np.float32)
        vector /= np.linalg.norm(vector)
        candidates.append((f"p{i}", 0.9 - 0.05 * i, vector))
    query = candidates[0][2]
    selected, report = MaximalMarginalRelevance().select(query, candidates, k=3)
    assert selected[0][0] == "p0"                 # relevance still leads
    assert report.lambda_used >= 0.6              # little diversification needed


@pytest.mark.asyncio
async def test_search_returns_a_calibrated_confidence_block(node):
    await node.remember("Coolant pressure below 1.8 bar is a hard stop", collection="semantic")
    result = await node.pipeline.search("coolant pressure", k=3)
    assert "guarantee" in result.confidence
    assert result.confidence["set_size"] >= 0
