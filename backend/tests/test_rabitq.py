"""RaBitQ: the estimator must be unbiased and the bound must actually hold."""
from __future__ import annotations

import numpy as np
import pytest

from aegis.memory.quantize import BinaryQuantizer
from aegis.memory.rabitq import (ColdCodebook, RaBitQ, Rotation, adaptive_shortlist, fwht)


def unit(rng, n, d, clusters=20, spread=0.5):
    centres = rng.normal(size=(clusters, d))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    data = centres[rng.integers(0, clusters, n)] + rng.normal(size=(n, d)) * spread
    return (data / np.linalg.norm(data, axis=1, keepdims=True)).astype(np.float32)


def test_fwht_is_orthogonal_up_to_scale():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(8, 64)).astype(np.float32)
    y = fwht(x) / np.sqrt(64)
    assert np.allclose(np.linalg.norm(x, axis=1), np.linalg.norm(y, axis=1), atol=1e-3)


def test_fwht_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        fwht(np.zeros((1, 100), dtype=np.float32))


@pytest.mark.parametrize("dim", [64, 100])
def test_rotation_preserves_inner_products(dim):
    """Fast and dense paths must both be genuinely orthogonal."""
    rng = np.random.default_rng(1)
    rotation = Rotation(dim)
    a, b = rng.normal(size=dim).astype(np.float32), rng.normal(size=dim).astype(np.float32)
    assert np.isclose(a @ b, rotation.apply(a) @ rotation.apply(b), rtol=1e-3, atol=1e-3)
    assert rotation.fast is ((dim & (dim - 1)) == 0)


def test_estimator_is_unbiased_and_bound_holds():
    rng = np.random.default_rng(2)
    data = unit(rng, 4000, 128)
    codec = RaBitQ(128)
    codec.fit(data)
    codes = codec.encode(data)

    errors, covered = [], []
    for query in unit(rng, 20, 128):
        estimate, bound = codec.estimate(codec.prepare(query), codes)
        error = estimate - data @ query
        errors.append(error)
        covered.append(np.abs(error) <= bound)

    errors = np.concatenate(errors)
    # Unbiased: the mean error is centred on zero, not merely small.
    assert abs(float(errors.mean())) < 0.005, f"bias {errors.mean()}"
    # The stated confidence is 1 - 2exp(-eps^2/2); hold it to a little under.
    assert float(np.mean(np.concatenate(covered))) > 0.95


def test_beats_plain_sign_bits_on_recall():
    """The whole reason for the extra eight bytes per vector."""
    rng = np.random.default_rng(3)
    data = unit(rng, 3000, 128)
    codec = RaBitQ(128)
    codec.fit(data)
    codes = codec.encode(data)
    sign = BinaryQuantizer.encode(data)

    k, rabit, plain = 10, [], []
    for query in unit(rng, 25, 128):
        gold = set(np.argsort(-(data @ query))[:k].tolist())
        estimate, _ = codec.estimate(codec.prepare(query), codes)
        rabit.append(len(gold & set(np.argsort(-estimate)[:k * 6].tolist())) / k)
        approx = BinaryQuantizer.similarity(np.packbits(query > 0).reshape(1, -1), sign, 128)
        plain.append(len(gold & set(np.argsort(-approx)[:k * 6].tolist())) / k)
    assert np.mean(rabit) > np.mean(plain), (np.mean(rabit), np.mean(plain))


def test_adaptive_shortlist_contains_the_true_top_k():
    rng = np.random.default_rng(4)
    data = unit(rng, 2000, 128)
    codec = RaBitQ(128)
    codec.fit(data)
    codes = codec.encode(data)
    for query in unit(rng, 10, 128):
        estimate, bound = codec.estimate(codec.prepare(query), codes)
        shortlist = set(adaptive_shortlist(estimate, bound, 10).tolist())
        assert set(np.argsort(-(data @ query))[:10].tolist()) <= shortlist


def test_adaptive_shortlist_respects_a_budget():
    rng = np.random.default_rng(5)
    estimate = rng.normal(size=500).astype(np.float32)
    bound = np.full(500, 10.0, dtype=np.float32)      # bounds so wide nothing prunes
    assert adaptive_shortlist(estimate, bound, 10).size == 500
    assert adaptive_shortlist(estimate, bound, 10, budget=64).size == 64


def test_codebook_append_and_swap_remove_keep_identity():
    rng = np.random.default_rng(6)
    data = unit(rng, 200, 64)
    book = ColdCodebook(RaBitQ(64))
    for i, vector in enumerate(data):
        book.put(f"p{i}", vector)
    assert len(book) == 200

    assert book.pop("p0") and book.pop("p199") and book.pop("p100")
    assert not book.pop("p0")
    assert len(book) == 197
    assert len(set(book.ids())) == 197
    assert "p0" not in book and "p100" not in book and "p50" in book

    ids, codes = book.subset({"p50", "p51", "missing"})
    assert sorted(ids) == ["p50", "p51"] and len(codes) == 2

    # Every surviving id must still decode to its own vector, not a neighbour's.
    codec = book.codec
    for point_id in ("p50", "p150", "p198"):
        ids, codes = book.subset({point_id})
        index = int(point_id[1:])
        estimate, _ = codec.estimate(codec.prepare(data[index]), codes)
        assert estimate[0] > 0.9, (point_id, estimate[0])


def test_codes_are_views_not_copies():
    """The cold scan must not allocate the corpus on every query."""
    book = ColdCodebook(RaBitQ(64))
    rng = np.random.default_rng(7)
    for i, vector in enumerate(unit(rng, 50, 64)):
        book.put(f"p{i}", vector)
    first, second = book.codes(), book.codes()
    assert first.bits.base is second.bits.base
