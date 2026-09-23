"""SLO accounting and the degradation ladder."""
from __future__ import annotations

import pytest

from aegis.core.bus import EventBus
from aegis.core.slo import FEATURES, Level, SLOManager


@pytest.fixture
def slo() -> SLOManager:
    manager = SLOManager(EventBus(), latency_target_ms=100.0)
    manager.MIN_DWELL_S = 0.0
    return manager


def test_healthy_traffic_stays_at_full(slo):
    for _ in range(40):
        slo.observe(40.0, True)
    assert slo.evaluate() is Level.FULL
    assert slo.disabled() == []


def test_sustained_breach_walks_down_the_ladder(slo):
    for _ in range(60):
        slo.observe(400.0, True)
    seen = [slo.evaluate() for _ in range(4)]
    assert seen == [Level.ECONOMISE, Level.TRIM, Level.ESSENTIAL, Level.SURVIVAL]
    assert not slo.allows("cross_encoder")
    assert not slo.allows("late_interaction")


def test_recovery_climbs_back_up(slo):
    for _ in range(60):
        slo.observe(400.0, True)
    for _ in range(4):
        slo.evaluate()
    assert slo.level is Level.SURVIVAL
    slo.latencies.clear()
    slo.outcomes.clear()
    for _ in range(40):
        slo.observe(20.0, True)
    for _ in range(5):
        slo.evaluate()
    assert slo.level is Level.FULL


def test_hysteresis_prevents_flapping():
    manager = SLOManager(EventBus(), latency_target_ms=100.0)   # default dwell applies
    for _ in range(60):
        manager.observe(500.0, True)
    first = manager.evaluate()
    second = manager.evaluate()
    assert first is second                       # cannot move twice inside the dwell window


def test_scheduler_pressure_alone_can_shed(slo):
    for _ in range(40):
        slo.observe(10.0, True)
    assert slo.evaluate(pressure=0.95) is Level.ECONOMISE


def test_failures_count_against_the_budget(slo):
    for _ in range(40):
        slo.observe(10.0, ok=False)
    assert slo.burn_rate > 0
    assert slo.success_rate == 0.0


def test_every_feature_names_a_rung():
    assert set(FEATURES.values()) <= set(Level)
    assert FEATURES["triton_escalation"] is Level.ECONOMISE
    assert FEATURES["sparse"] is Level.SURVIVAL


def test_manual_override_pins_the_level(slo):
    slo.override(Level.ESSENTIAL)
    for _ in range(40):
        slo.observe(5.0, True)
    assert slo.evaluate() is Level.ESSENTIAL      # healthy traffic cannot lift a pin
    slo.override(None)
    for _ in range(40):
        slo.observe(5.0, True)
    assert slo.evaluate().value < Level.ESSENTIAL.value


@pytest.mark.asyncio
async def test_degraded_node_still_answers_and_says_so(node):
    await node.remember("Coolant pressure below 1.8 bar is a hard stop", collection="semantic")
    node.slo.override(Level.ESSENTIAL)
    result = await node.pipeline.search("coolant pressure", k=3)
    assert result.results                                    # still answering
    assert result.degradation["level"] == "ESSENTIAL"
    assert "cross_encoder" in result.degradation["disabled"]
    assert "maxsim_ms" not in result.stages                  # the expensive stage was shed


@pytest.mark.asyncio
async def test_survival_mode_narrows_the_fetch(node):
    for i in range(12):
        await node.remember(f"observation {i} about the conveyor")
    full = await node.pipeline.search("conveyor", k=3)
    node.slo.override(Level.SURVIVAL)
    survival = await node.pipeline.search("conveyor", k=3)
    assert survival.results
    assert survival.degradation["level"] == "SURVIVAL"
    assert full.stages.get("rerank_ms") is not None
