"""The on-device adaptations, and the gate that refuses the ones that do not help."""
from __future__ import annotations

import numpy as np
import pytest

from aegis.inference.adaptation import AdaptationGate, ProbeSet, paired_bootstrap
from aegis.inference.geometry import (AnisotropyProbe, CorpusGeometry,
                                      ledoit_wolf_shrinkage)
from aegis.inference.lexicon import TokenLexicon

WORDS = ("conveyor bearing vibration coolant pressure interlock gantry spindle torque "
         "encoder chiller hydraulic gearbox tensioner runout backlash servo relief "
         "housing shift maintenance replaced logged reading threshold alarm").split()


def corpus(rng, n=1200, length=12):
    return [" ".join(rng.choice(WORDS, size=length)) for _ in range(n)]


def anisotropic(rng, n, d, cone=3.0):
    """Vectors in a narrow cone — what a pretrained space actually looks like."""
    direction = rng.normal(size=d)
    direction /= np.linalg.norm(direction)
    data = rng.normal(size=(n, d)) / np.sqrt(d) + cone * direction
    return (data / np.linalg.norm(data, axis=1, keepdims=True)).astype(np.float32)


# -- geometry ------------------------------------------------------------

def test_probe_detects_and_whitening_removes_anisotropy():
    rng = np.random.default_rng(0)
    data = anisotropic(rng, 4000, 64)
    before = AnisotropyProbe.measure(data)
    assert before["mean_random_pair_cosine"] > 0.3
    assert "anisotropic" in before["verdict"]

    geometry = CorpusGeometry(64)
    geometry.observe(data)
    assert geometry.fit() is not None
    after = AnisotropyProbe.measure(geometry.transform(data))
    assert abs(after["mean_random_pair_cosine"]) < 0.05


def test_shrinkage_falls_as_evidence_grows():
    """The Ledoit-Wolf intensity must respond to the real sample size.

    Passing a subsample's own length here instead of the true count drives the
    intensity to 1, which silently replaces whitening with a bare rotation —
    a bug this repository shipped once and must not ship twice.
    """
    rng = np.random.default_rng(1)
    d = 32
    base = rng.normal(size=(d, d))
    covariance = base @ base.T / d
    sample = rng.multivariate_normal(np.zeros(d), covariance, size=2000)
    small = ledoit_wolf_shrinkage(sample, covariance, n_total=200)
    large = ledoit_wolf_shrinkage(sample, covariance, n_total=200_000)
    assert 0.0 <= large < small <= 1.0


def test_rank_selection_refuses_to_amplify_noise():
    """A near-rank-deficient space must not be whitened at full dimension."""
    rng = np.random.default_rng(2)
    latent = rng.normal(size=(3000, 8))
    basis = rng.normal(size=(8, 128))
    data = (latent @ basis + rng.normal(size=(3000, 128)) * 1e-4).astype(np.float32)
    geometry = CorpusGeometry(128, energy_target=0.99)
    geometry.observe(data)
    version = geometry.fit()
    assert version is not None
    assert geometry.rank < 32, f"kept {geometry.rank} of 128 near-empty directions"
    assert geometry.transform(data).shape[1] == geometry.rank


def test_geometry_refuses_to_fit_on_too_little_data():
    geometry = CorpusGeometry(64)
    geometry.observe(np.random.default_rng(3).normal(size=(10, 64)))
    assert geometry.fit() is None
    assert not geometry.fitted
    assert not geometry.enable(True)          # cannot arm what was never fitted


def test_geometry_round_trips_through_disk(tmp_path):
    rng = np.random.default_rng(4)
    data = anisotropic(rng, 2000, 64)
    geometry = CorpusGeometry(64)
    geometry.observe(data)
    version = geometry.fit()
    assert geometry.save(tmp_path / "geo.npz")

    restored = CorpusGeometry(64)
    assert restored.load(tmp_path / "geo.npz").id == version.id
    assert np.allclose(restored.transform(data[:20]), geometry.transform(data[:20]))
    # A transform fitted for another dimension must be refused, not reshaped.
    assert CorpusGeometry(32).load(tmp_path / "geo.npz") is None


def test_nested_ladder_is_monotone_in_energy():
    rng = np.random.default_rng(5)
    geometry = CorpusGeometry(64)
    geometry.observe(anisotropic(rng, 2000, 64))
    geometry.fit()
    energies = [rung["energy_retained"] for rung in geometry.nested_ladder()]
    assert energies == sorted(energies)


# -- lexicon -------------------------------------------------------------

def test_lexicon_is_exactly_the_mask_until_it_has_evidence():
    lexicon = TokenLexicon(1000, min_observations=10_000)
    ids = np.array([[5, 9, 0], [7, 0, 0]], dtype=np.int64)
    mask = np.array([[1, 1, 0], [1, 0, 0]], dtype=np.float32)
    assert np.array_equal(lexicon.weights(ids, mask), mask)
    assert not lexicon.informed


def test_lexicon_damps_frequent_tokens_and_never_weights_padding():
    lexicon = TokenLexicon(100, a=1e-3, min_observations=10)
    lexicon.observe([[1] * 500 + [2]])            # token 1 common, token 2 rare
    assert lexicon.informed
    ids = np.array([[1, 2, 0]], dtype=np.int64)
    mask = np.array([[1.0, 1.0, 0.0]], dtype=np.float32)
    weights = lexicon.weights(ids, mask)
    assert weights[0, 0] < weights[0, 1]          # common token damped
    assert weights[0, 2] == 0.0                   # padding never contributes


def test_lexicon_survives_an_all_stopword_document():
    lexicon = TokenLexicon(100, a=1e-9, min_observations=1)
    lexicon.observe([[1] * 1000])
    ids = np.array([[1, 1]], dtype=np.int64)
    mask = np.array([[1.0, 1.0]], dtype=np.float32)
    assert lexicon.weights(ids, mask).sum() > 0    # must still produce a vector


def test_corrupt_lexicon_is_a_cache_miss_not_a_boot_failure(tmp_path):
    path = tmp_path / "lex.npz"
    path.write_bytes(b"not an npz file at all")
    assert TokenLexicon.load(path) is None
    assert TokenLexicon.load(tmp_path / "absent.npz") is None


# -- the gate ------------------------------------------------------------

def test_bootstrap_separates_a_real_effect_from_noise():
    rng = np.random.default_rng(6)
    baseline = rng.random(400)
    assert paired_bootstrap(baseline, baseline.copy())["low"] <= 0.0
    lifted = np.clip(baseline + 0.15, 0, 1)
    assert paired_bootstrap(baseline, lifted)["low"] > 0.0


def test_gate_refuses_a_candidate_that_does_nothing():
    rng = np.random.default_rng(7)
    texts = corpus(rng)
    space = {t: rng.normal(size=32).astype(np.float32) for t in set(texts)}

    def encode(batch):
        out = np.vstack([space.setdefault(t, rng.normal(size=32).astype(np.float32))
                         for t in batch])
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    gate = AdaptationGate(min_documents=200, min_probes=50)
    report = gate.evaluate(texts, encode, [("identity", lambda v: v)], queries=120)
    assert report["ran"]
    assert report["decision"]["adopted"] is False
    assert gate.adopted is None


def test_gate_adopts_a_candidate_that_genuinely_helps():
    """A transform that recovers the signal a noisy encoder buried must win."""
    rng = np.random.default_rng(8)
    texts = corpus(rng, n=700)
    # A bag-of-words encoder, so a paraphrase (a subset of the words) really is
    # near its source. The clean 16-d signal is then hidden behind 64 noisy
    # dimensions whose variance dwarfs it, which is exactly the situation
    # PCA-whitening exists to undo.
    word_vectors = {w: rng.normal(size=16).astype(np.float32) for w in WORDS}
    mix = rng.normal(size=(16, 64)).astype(np.float32)

    def encode(batch):
        signal = np.vstack([
            np.sum([word_vectors[w] for w in text.split() if w in word_vectors]
                   or [np.zeros(16, np.float32)], axis=0)
            for text in batch]) @ mix
        noisy = signal + rng.normal(size=(len(batch), 64)).astype(np.float32) * 0.9
        return (noisy / np.linalg.norm(noisy, axis=1, keepdims=True)).astype(np.float32)

    probe_vectors = encode(texts[:600])
    geometry = CorpusGeometry(64, energy_target=0.95)
    geometry.observe(probe_vectors)
    assert geometry.fit() is not None

    gate = AdaptationGate(min_documents=200, min_probes=50, margin=0.001)
    report = gate.evaluate(texts, encode, [("whitening", geometry.transform)], queries=200)
    assert report["ran"]
    result = next(r for r in report["results"] if r["candidate"] == "whitening")
    assert result["delta"] > 0 and result["significant"], result
    assert report["decision"]["adopted"] is True
    assert gate.adopted == "whitening"


def test_gate_will_not_run_on_a_corpus_too_small_to_judge():
    gate = AdaptationGate(min_documents=500)
    report = gate.evaluate(["short text here"] * 10, lambda b: np.zeros((len(b), 4)), [])
    assert report["ran"] is False and "too small" in report["reason"]


def test_probe_set_ground_truth_is_the_source_document():
    rng = np.random.default_rng(9)
    probes = ProbeSet.build(corpus(rng, n=300), queries=80)
    assert probes is not None and len(probes) == 80
    for query, gold in zip(probes.queries, probes.truth):
        source = set(probes.documents[gold].split())
        assert set(query.split()) <= source        # a probe is a subset of its source


def test_probe_set_refuses_a_corpus_with_nothing_to_probe():
    assert ProbeSet.build(["tiny"] * 10, queries=50) is None
