"""Energy accounting and the provenance receipt."""
from __future__ import annotations

import json

import pytest

from aegis.core.energy import EnergyMeter, Source
from aegis.core.provenance import Provenance, merkle_root


# -- energy ---------------------------------------------------------------

def test_a_modelled_reading_never_claims_to_be_measured():
    """The whole point of the source ladder.

    RAPL and a battery are measurements; CPU-time times an assumed wattage is
    a guess about silicon this code has never seen. A guess presented as a
    measurement survives exactly until somebody checks it.
    """
    meter = EnergyMeter(watts_per_busy_core=6.0)
    with meter.measure("query") as span:
        sum(i * i for i in range(200_000))
    reading = span.reading
    assert reading.joules > 0
    if reading.source is Source.MODEL:
        assert reading.as_dict()["measured"] is False
        assert "not a measurement" in meter.snapshot()["how"]
    else:
        assert reading.source.measured


def test_joules_track_work_done():
    meter = EnergyMeter(watts_per_busy_core=6.0)
    with meter.measure("small"):
        sum(i for i in range(50_000))
    with meter.measure("large"):
        sum(i * i for i in range(2_000_000))
    small = meter.accounts["small"].joules
    large = meter.accounts["large"].joules
    assert large > small, (small, large)


def test_answers_per_battery_percent_needs_a_real_battery():
    """It returns None rather than inventing a pack that is not there."""
    blind = EnergyMeter(battery_capacity_wh=None)
    with blind.measure("query"):
        sum(i for i in range(100_000))
    if blind.battery_capacity_wh is None:
        assert blind.per_battery_percent("query") is None

    known = EnergyMeter(battery_capacity_wh=50.0)
    with known.measure("query"):
        sum(i * i for i in range(300_000))
    fits = known.per_battery_percent("query")
    assert fits is not None and fits > 0
    # Sanity: 1% of 50 Wh is 1,800 J, so the count must equal 1800/J-per-op.
    per_op = known.accounts["query"].joules / known.accounts["query"].ops
    assert fits == pytest.approx(1800.0 / per_op, rel=1e-6)


def test_unknown_operation_kinds_do_not_explode():
    meter = EnergyMeter()
    assert meter.per_battery_percent("never-measured") is None
    assert meter.snapshot()["by_kind"] == {}


# -- provenance -----------------------------------------------------------

def test_merkle_root_is_order_independent_and_change_sensitive():
    a = merkle_root([("x", "1"), ("y", "2")])
    assert a == merkle_root([("y", "2"), ("x", "1")])
    assert a != merkle_root([("x", "1"), ("y", "3")])
    assert merkle_root([]) == merkle_root([])


def test_receipt_covers_source_and_models(tmp_path):
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    receipt = Provenance(root).receipt()
    assert receipt["source"]["files"] > 50
    assert len(receipt["root"]) == 64
    assert receipt["git"]["commit"] is None or len(receipt["git"]["commit"]) == 40
    # It must say what it does *not* prove.
    assert "integrity, not authenticity" in receipt["claims"]


def test_verify_locates_a_mismatch_rather_than_merely_reporting_one():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    prov = Provenance(root)
    mine = prov.full()

    same = prov.verify_against({"root": mine["root"], "files": mine["files"]})
    assert same["match"] is True

    tampered = dict(mine["files"])
    victim = sorted(tampered)[0]
    tampered[victim] = "0" * 64
    tampered["ghost.py"] = "1" * 64
    out = prov.verify_against({"root": "deadbeef", "files": tampered})
    assert out["match"] is False
    assert victim in out["changed_files"]
    assert "ghost.py" in out["removed_files"]


def test_a_dirty_tree_is_reported_not_hidden():
    """A receipt that concealed uncommitted changes would be worthless."""
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    git = Provenance(root).receipt()["git"]
    assert "clean" in git and "dirty_files" in git
