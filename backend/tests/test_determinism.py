"""The property the whole simulator rests on.

If an execution is not a pure function of its seed, then a counterexample
cannot be replayed, a shrinker cannot ask "does it still fail without this
step?", and the entire approach collapses into ordinary flaky testing. These
tests defend that property directly.
"""
import asyncio

import pytest

from aegis.core import determinism
from aegis.core.determinism import simulated
from aegis.sim.world import run_seed


def test_time_only_moves_when_the_simulation_moves_it():
    with simulated(7) as world:
        first = determinism.now()
        assert determinism.now() == first          # real time passing changes nothing
        world.advance(30.0)
        assert determinism.now() == pytest.approx(first + 30.0)


def test_a_clock_may_be_made_to_go_backwards():
    """Devices do this. A simulator that cannot express it cannot find the bug."""
    with simulated(7) as world:
        before = determinism.now()
        world.skew(-120.0)
        assert determinism.now() < before


def test_sleeping_advances_the_clock_without_waiting():
    with simulated(7) as world:
        start = determinism.now()
        asyncio.run(determinism.sleep(3600.0))
        assert determinism.now() == pytest.approx(start + 3600.0)
        assert world.now() == pytest.approx(start + 3600.0)


def test_the_environment_is_restored_afterwards():
    real = determinism.now()
    with simulated(7):
        assert determinism.now() != real
    assert determinism.now() >= real               # back to the wall clock


def test_the_generator_is_seeded_inside_the_simulation():
    with simulated(11):
        first = [determinism.rng().random() for _ in range(5)]
    with simulated(11):
        assert [determinism.rng().random() for _ in range(5)] == first


def test_the_same_seed_produces_the_same_history():
    a_violation, a = run_seed(3, steps=80)
    b_violation, b = run_seed(3, steps=80)
    assert (a_violation is None) == (b_violation is None)
    assert a["operations"] == b["operations"]
    assert a["messages"] == b["messages"]
    assert a["steps"] == b["steps"]


def test_different_seeds_produce_different_histories():
    """Otherwise the sweep is one execution run three thousand times."""
    shapes = {(run_seed(s, steps=80)[1]["operations"],
               run_seed(s, steps=80)[1]["messages"]) for s in range(6)}
    assert len(shapes) > 1


def test_a_recorded_history_replays_exactly():
    from aegis.sim.world import Simulation
    with simulated(19) as world:
        first = Simulation(19, world=world)
        asyncio.run(first.run(60))
        history = list(first.history)
    with simulated(19) as world:
        again = Simulation(19, world=world)
        asyncio.run(again.run(60, replay=history))
    assert [a for a, _ in again.history] == [a for a, _ in history]


def test_identifiers_are_part_of_the_execution():
    """The leak that made a 500-seed sweep return 454 failures, then 456.

    An operation id is what the IBLT hashes and what a fetch is sorted by, so
    ids drawn from the wall clock make the *reconciliation* non-deterministic
    while everything else replays perfectly — which looks exactly like noise.
    """
    from aegis.core.ids import ulid
    with simulated(101):
        first = [ulid() for _ in range(8)]
    with simulated(101):
        assert [ulid() for _ in range(8)] == first
    assert len({ulid() for _ in range(2000)}) == 2000      # still unique outside


def test_a_sweep_returns_the_same_count_twice():
    """The property the leak above broke, asserted end to end."""
    def sweep():
        return sum(run_seed(s, steps=120, signed=False)[0] is not None for s in range(12))
    assert sweep() == sweep()


def test_the_audit_forbids_reaching_past_the_environment():
    """The lint is part of the guarantee, so the suite runs it."""
    import subprocess
    import sys
    from pathlib import Path
    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_determinism.py"
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout


def test_signing_closes_what_the_unsigned_mesh_leaves_open():
    """The headline claim, as a test rather than a paragraph.

    Six seeds is enough to be a regression test without being a sweep; the
    full sweep lives in scripts/simulate.py.
    """
    unsigned = [run_seed(s, steps=150, signed=False)[0] for s in range(6)]
    signed = [run_seed(s, steps=150, signed=True)[0] for s in range(6)]
    assert any(v is not None and v.invariant == "bodies-intact" for v in unsigned)
    assert all(v is None for v in signed), [v.detail for v in signed if v]
