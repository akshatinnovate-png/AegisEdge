"""ANN: graph recall, quantizer calibration, strategy selection, cold tier."""
from __future__ import annotations

import numpy as np
import pytest

from aegis.memory.ann import AdaptiveVectorIndex, CostModel, Strategy
from aegis.memory.hnsw import HnswIndex, HnswParams
from aegis.memory.pq import IvfPqIndex, PqParams, ProductQuantizer


def _corpus(n: int, dim: int, clustered: bool = True, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if clustered:
        centres = rng.normal(size=(max(n // 100, 4), dim))
        data = np.vstack([c + 0.3 * rng.normal(size=(n // len(centres) + 1, dim)) for c in centres])[:n]
    else:
        data = rng.normal(size=(n, dim))
    data = data.astype(np.float32)
    return data / np.linalg.norm(data, axis=1, keepdims=True)


def _recall(index, data: np.ndarray, k: int = 10, probes: int = 25) -> float:
    hits = 0
    for i in range(0, len(data), max(1, len(data) // probes)):
        truth = {int(j) for j in np.argsort(-(data @ data[i]))[:k]}
        found = {int(pid[1:]) for pid, _ in index.search(data[i], k)}
        hits += len(truth & found)
    counted = len(range(0, len(data), max(1, len(data) // probes)))
    return hits / (counted * k)


def test_hnsw_beats_90_percent_recall():
    data = _corpus(1500, 64)
    index = HnswIndex(64, HnswParams(ef_construction=96))
    for i, vector in enumerate(data):
        index.add(f"p{i}", vector)
    assert _recall(index, data) >= 0.90
    assert index.snapshot()["layers"] >= 2                 # it actually built a hierarchy


def test_hnsw_touches_far_fewer_points_than_a_scan():
    data = _corpus(2000, 64)
    index = HnswIndex(64, HnswParams())
    for i, vector in enumerate(data):
        index.add(f"p{i}", vector)
    index.stats.distance_evals = 0
    index.stats.searches = 0
    for i in range(0, 2000, 200):
        index.search(data[i], 10)
    per_query = index.stats.distance_evals / index.stats.searches
    assert per_query < len(data) * 0.75                    # the graph is doing real work


def test_hnsw_soft_delete_keeps_the_graph_navigable():
    data = _corpus(400, 32)
    index = HnswIndex(32, HnswParams())
    for i, vector in enumerate(data):
        index.add(f"p{i}", vector)
    for i in range(0, 200):
        index.remove(f"p{i}")
    assert index.live == 200
    found = index.search(data[350], 5)
    assert found[0][0] == "p350"
    assert all(not pid.startswith("p1") or int(pid[1:]) >= 200 for pid, _ in found)


def test_opq_rotation_reduces_quantization_error():
    data = _corpus(3000, 64)
    plain = ProductQuantizer(64, PqParams(subspaces=8, opq_iterations=0)).train(data)
    rotated = ProductQuantizer(64, PqParams(subspaces=8, opq_iterations=5)).train(data)
    assert rotated.residual_error <= plain.residual_error
    assert rotated.compression >= 30                       # 8 bytes for a 64-dim vector


def test_ivfpq_calibrates_nprobe_and_hits_target_recall():
    data = _corpus(2500, 64)
    index = IvfPqIndex(64, lists=24, nprobe=2, params=PqParams(subspaces=8))
    index.train(data)
    for i, vector in enumerate(data):
        index.add(f"p{i}", vector)
    assert index.calibration["nprobe"] >= 2
    assert index.calibrated_depth > 0
    assert _recall(index, data) >= 0.90


def test_uniform_data_is_told_to_probe_more_than_clustered():
    """The calibration is data-dependent, which is the entire point of it."""
    clustered = IvfPqIndex(64, lists=20, params=PqParams(subspaces=8)).train(_corpus(2000, 64, True))
    uniform = IvfPqIndex(64, lists=20, params=PqParams(subspaces=8)).train(_corpus(2000, 64, False))
    assert uniform.calibration["probe_fraction"] > clustered.calibration["probe_fraction"]


def test_cost_model_calibrates_on_this_machine():
    cost = CostModel().calibrate(dim=64, sample=1024)
    assert cost.calibrated
    assert cost.flat_ns_per_point > 0
    assert cost.hnsw_crossover >= 5_000
    assert cost.choose(100) is Strategy.FLAT
    assert cost.choose(cost.ivf_crossover + 1) is Strategy.IVF_PQ
    assert cost.choose(5_000, memory_pressure=0.95) is Strategy.IVF_PQ   # RAM binds first


def test_adaptive_index_supports_prefiltered_exact_search():
    cost = CostModel().calibrate(dim=32, sample=512)
    index = AdaptiveVectorIndex(32, cost, "episodic")
    data = _corpus(300, 32)
    for i, vector in enumerate(data):
        index.add(f"p{i}", vector)
    allow = {"p7", "p11", "p250"}
    found = index.search(data[7], 3, allow=allow)
    assert {pid for pid, _ in found} <= allow
    assert found[0][0] == "p7"


@pytest.mark.parametrize("strategy", [Strategy.HNSW, Strategy.IVF_PQ])
def test_forced_strategies_stay_accurate(strategy):
    cost = CostModel().calibrate(dim=64, sample=512)
    index = AdaptiveVectorIndex(64, cost, "bench")
    data = _corpus(1200, 64)
    for i, vector in enumerate(data):
        index.add(f"p{i}", vector)
    index.force(strategy)
    assert index.strategy is strategy
    assert _recall(index, data, k=10, probes=20) >= 0.85


def test_cost_model_does_not_choose_a_graph_where_exhaustive_search_wins():
    """The crossover was measured, not assumed, and the measurement was brutal.

    `scripts/strategy_bakeoff.py` forced each strategy onto the same corpus:

        20,000 points   flat    p50 0.886 ms   recall 1.000   build   0 s
                        hnsw    p50 4.340 ms   recall 0.773   build 252 s
                        ivf_pq  p50 550.5 ms   recall 0.997   build 123 s

    Exhaustive search was 4.9x faster than the graph, exact where the graph
    lost a quarter of its recall, and free to build - and the cost model was
    selecting the graph from 5,000 points upward, because it timed a hop as
    one vectorised numpy call and so missed the interpreter overhead that
    dominates a real traversal.
    """
    from aegis.memory.ann import CostModel, Strategy

    cost = CostModel().calibrate(dim=256, sample=2048)

    # Timing a hop honestly must show it costs multiples of a scanned point.
    assert cost.detail["interpreter_overhead_x"] > 4.0
    # And the calibration must be reproducible: a single timing swung this
    # between 40,000 and 110,000 across consecutive runs on one machine.
    again = CostModel().calibrate(dim=256, sample=2048)
    ratio = again.hnsw_crossover / max(cost.hnsw_crossover, 1)
    assert 0.5 < ratio < 2.0, (cost.hnsw_crossover, again.hnsw_crossover)

    # The bake-off measured flat as both faster and exact at every scale it
    # tested, the largest being 20,000 points. Assert the range it covers,
    # with headroom — not a specific crossover, which is a timing-derived
    # number and would make this a test of the machine's mood.
    for count in (1_000, 5_000, 20_000):
        assert cost.choose(count) is Strategy.FLAT, (
            f"chose {cost.choose(count).value} at {count:,} points, where the "
            "bake-off shows exhaustive search is both faster and exact")

    # And the crossover must land well past anything an edge device holds,
    # rather than on top of it.
    assert cost.hnsw_crossover > 50_000, cost.as_dict()

    # It must still switch eventually, or it is not a cost model at all.
    assert cost.choose(cost.hnsw_crossover * 2) is not Strategy.FLAT

    # IVF-PQ costs 550 ms a query here: it is a memory decision, never a
    # latency one, and must not be reachable on size alone at edge scale.
    assert cost.choose(min(100_000, cost.hnsw_crossover - 1)) is not Strategy.IVF_PQ
    assert cost.choose(20_000, memory_pressure=0.95) is Strategy.IVF_PQ
