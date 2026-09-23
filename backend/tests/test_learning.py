"""On-device adapter, differential privacy, secure aggregation."""
from __future__ import annotations

import numpy as np
import pytest

from aegis.learning.adapter import RetrievalAdapter, TrainingExample
from aegis.learning.federated import (FederatedClient, FederatedCoordinator,
                                      pairwise_mask, required_cohort)
from aegis.learning.privacy import GaussianMechanism, PrivacyBudget

DIM = 48


def _unit(rng) -> np.ndarray:
    vector = rng.normal(size=DIM).astype(np.float32)
    return vector / np.linalg.norm(vector)


def test_adapter_learns_to_reorder_from_feedback():
    rng = np.random.default_rng(0)
    query, good, bad = _unit(rng), _unit(rng), _unit(rng)
    adapter = RetrievalAdapter(DIM, rank=8)
    base = [("good", float(query @ good), good), ("bad", float(query @ bad), bad)]
    for _ in range(80):
        adapter.learn([TrainingExample(query=query, positive=good, negative=bad)])
    ranked = [pid for pid, _, _ in adapter.rescore(query, base)]
    assert ranked[0] == "good"
    assert adapter.stats.last_loss == 0.0
    assert adapter.version > 0


def test_adapter_is_small_enough_to_live_on_a_device():
    adapter = RetrievalAdapter(384, rank=16)
    assert adapter.snapshot()["bytes"] < 100_000          # ~49 KB, not a fine-tuned model


def test_adapter_persists_and_reloads(tmp_path):
    rng = np.random.default_rng(1)
    adapter = RetrievalAdapter(DIM, rank=4)
    adapter.learn([TrainingExample(query=_unit(rng), positive=_unit(rng))])
    adapter.save(tmp_path / "adapter.json")
    restored = RetrievalAdapter(DIM, rank=4)
    assert restored.load(tmp_path / "adapter.json")
    assert np.allclose(restored.u, adapter.u)


def test_adapter_refuses_a_mismatched_checkpoint(tmp_path):
    RetrievalAdapter(DIM, rank=4).save(tmp_path / "a.json")
    assert not RetrievalAdapter(DIM, rank=16).load(tmp_path / "a.json")


def test_gaussian_mechanism_clips_then_noises():
    mechanism = GaussianMechanism(clip_norm=1.0)
    update = np.full(64, 5.0, dtype=np.float32)
    noisy, info = mechanism.privatize(update, epsilon=1.0, delta=1e-5)
    assert info["clipped_from"] > 1.0
    assert info["sigma"] > 0
    assert not np.allclose(noisy, update)


def test_privacy_budget_stops_contributing_when_spent():
    budget = PrivacyBudget(epsilon_total=1.0)
    assert budget.spend(0.6)
    assert not budget.spend(0.6)                          # refuses to overspend
    assert budget.as_dict()["remaining"] == pytest.approx(0.4)


def test_pairwise_masks_cancel_exactly():
    cohort = [f"edge-{i}" for i in range(6)]
    total = np.zeros(256, dtype=np.float32)
    for me in cohort:
        for peer in cohort:
            if peer != me:
                mask = pairwise_mask(me, peer, 256, 3)
                total += mask if me < peer else -mask
    assert float(np.abs(total).max()) < 1e-5


def test_individual_contribution_hides_the_update():
    rng = np.random.default_rng(2)
    adapter = RetrievalAdapter(DIM, rank=4)
    adapter.learn([TrainingExample(query=_unit(rng), positive=_unit(rng))])
    cohort = ["edge-0", "edge-1", "edge-2"]
    client = FederatedClient("edge-0", adapter, PrivacyBudget())
    raw = adapter.diff_from(client.baseline)
    contribution = client.contribute(cohort, 1)
    assert contribution is not None
    assert float(np.linalg.norm(contribution.masked - raw)) > 1.0    # unrecognisable alone


def test_aggregation_reports_whether_the_cohort_is_large_enough():
    clients = []
    rng = np.random.default_rng(3)
    for i in range(5):
        adapter = RetrievalAdapter(DIM, rank=4, seed=1)
        adapter.learn([TrainingExample(query=_unit(rng), positive=_unit(rng))])
        clients.append(FederatedClient(f"edge-{i}", adapter, PrivacyBudget(epsilon_total=20.0)))
    cohort = [c.node_id for c in clients]
    coordinator = FederatedCoordinator(clients[0].adapter.parameters().size)
    result = coordinator.aggregate(cohort, [c.contribute(cohort, 1) for c in clients])
    assert result["status"] == "aggregated"
    assert result["cohort_for_usable_snr"] > len(clients)          # states what it would need
    assert result["usable"] is False
    assert "needs" in result["note"]


def test_disabling_dp_is_possible_and_reported():
    rng = np.random.default_rng(4)
    clients = []
    for i in range(3):
        adapter = RetrievalAdapter(DIM, rank=4, seed=1)
        adapter.learn([TrainingExample(query=_unit(rng), positive=_unit(rng))])
        clients.append(FederatedClient(f"edge-{i}", adapter, differential_privacy=False))
    cohort = [c.node_id for c in clients]
    coordinator = FederatedCoordinator(clients[0].adapter.parameters().size)
    result = coordinator.aggregate(cohort, [c.contribute(cohort, 1) for c in clients])
    assert result["usable"] is True
    assert clients[0].budget.spent == 0.0                          # no budget consumed


def test_round_is_abandoned_when_masks_would_not_cancel():
    rng = np.random.default_rng(5)
    clients = [FederatedClient(f"edge-{i}", RetrievalAdapter(DIM, rank=4, seed=1)) for i in range(6)]
    cohort = [c.node_id for c in clients]
    coordinator = FederatedCoordinator(clients[0].adapter.parameters().size)
    surviving = [c.contribute(cohort, 1) for c in clients[:3]]     # half the cohort vanished
    result = coordinator.aggregate(cohort, surviving)
    assert result["status"] == "abandoned"
    assert "uncancelled" in result["reason"]


def test_required_cohort_grows_with_noise():
    assert required_cohort(sigma=1.0, parameters=1024) < required_cohort(sigma=4.0, parameters=1024)


@pytest.mark.asyncio
async def test_feedback_endpoint_path_updates_the_adapter(node):
    point = await node.remember("Coolant pressure below 1.8 bar is a hard stop", collection="semantic")
    other = await node.remember("The canteen reopens at seven", collection="episodic")
    version = node.adapter.version
    for _ in range(node.settings.learning.batch):
        result = await node.feedback("coolant hard stop", point.id, other.id)
        assert result["accepted"]
    assert node.adapter.version > version
