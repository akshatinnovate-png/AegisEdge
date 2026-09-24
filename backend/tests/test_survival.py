"""Survival: the node under a storm of simultaneous faults.

Each test asserts an *invariant* rather than a happy path, because the claim
being made is not "it works" but "it does not lose data, leak across tenants,
or stop answering — whatever happens to it".
"""
from __future__ import annotations

import asyncio
import random

import pytest

from aegis.core.slo import Level


@pytest.mark.asyncio
async def test_node_keeps_answering_through_a_fault_storm(node):
    """Invariant: a query is always answered, or fails loudly — never hangs."""
    random.seed(5)
    for i in range(20):
        await node.remember(f"observation {i}: belt tension on line 2 drifting")

    faults = ["link_drop", "packet_loss", "latency_spike", "thermal_spike",
              "clock_skew", "disk_full", "corrupt_wal", "memory_pressure"]
    for fault in faults:
        await node.chaos.inject(fault, duration_s=0.2)

    answered = 0
    for i in range(12):
        result = await asyncio.wait_for(
            node.pipeline.search(f"belt tension {i % 4}", k=3), timeout=5.0)
        answered += int(bool(result.results))
    assert answered >= 10                         # degraded, never silent


@pytest.mark.asyncio
async def test_no_memory_is_lost_across_an_outage_and_corruption(node):
    """Invariant: every accepted write is either resident or reported lost."""
    ids = []
    for i in range(15):
        point = await node.remember(f"durable observation {i}")
        ids.append(point.id)
    node.store.archive()

    node.oracle.forced_offline = True
    await node.oracle.probe_once()
    for i in range(5):
        ids.append((await node.remember(f"offline observation {i}")).id)

    await node.chaos.inject("corrupt_segment", duration_s=0.2)
    result = await node.repair.scrub(repair=True)

    resident = {pid for pid in ids if pid in node.store.points}
    reported = set(result.get("repair", {}).get("unrecoverable", [])) if not result["clean"] else set()
    assert resident | reported >= set(ids) - set()   # accounted for, one way or the other
    assert len(resident) >= 15


@pytest.mark.asyncio
async def test_tenant_isolation_holds_under_load_and_degradation(node):
    """Invariant: no degradation level may leak another tenant's memories."""
    node.tenants.create("alpha")
    node.tenants.create("beta")
    for i in range(6):
        await node.remember(f"alpha secret {i}", tenant_id="alpha")
        await node.remember(f"beta secret {i}", tenant_id="beta")

    for level in (Level.FULL, Level.ECONOMISE, Level.TRIM, Level.ESSENTIAL, Level.SURVIVAL):
        node.slo.override(level)
        alpha = await node.pipeline.search("secret", k=10, tenant_id="alpha")
        beta = await node.pipeline.search("secret", k=10, tenant_id="beta")
        assert all("beta" not in hit["text"] for hit in alpha.results), level
        assert all("alpha" not in hit["text"] for hit in beta.results), level
    node.slo.override(None)


@pytest.mark.asyncio
async def test_restricted_memories_never_leave_under_any_fault(node):
    """Invariant: the policy boundary is not a fair-weather guarantee."""
    restricted = await node.remember("Operator 4471 jo@plant.io overrode the interlock")
    assert restricted.sync_class.value == "local_only"

    await node.chaos.inject("packet_loss", duration_s=0.2, rate=0.4)
    await node.chaos.inject("clock_skew", duration_s=0.2, ms=45_000)
    for _ in range(3):
        await node.sync.reconcile(trigger="storm")

    cloud = getattr(node.transport, "points", {})
    assert restricted.id not in cloud
    assert all(restricted.id != op.point_id for op in node.sync.queue.pending)


@pytest.mark.asyncio
async def test_wal_survives_garbage_appended_to_its_tail(node, settings):
    for i in range(8):
        await node.remember(f"observation {i}")
    await node.chaos.inject("corrupt_wal", duration_s=0.1, bytes_=128)

    from aegis.node import EdgeNode

    node.close()
    reopened = EdgeNode(settings)
    try:
        stats = reopened.store.recover()
        assert stats["applied"] >= 8
        assert stats["torn"] >= 1                # the garbage was seen and discarded
        assert len(reopened.store.points) >= 8
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_scheduler_sheds_background_work_rather_than_queueing_forever(node):
    from aegis.core.scheduler import Lane

    node.scheduler._depth[Lane.INTERACTIVE] = 20      # a crowd of waiting people
    with pytest.raises(RuntimeError, match="admission control"):
        await node.scheduler.submit("compact", lambda: asyncio.sleep(0), Lane.MAINTENANCE)
    assert node.scheduler.stats[Lane.MAINTENANCE].rejected >= 1


@pytest.mark.asyncio
async def test_clock_skew_does_not_break_convergence(node):
    for i in range(6):
        await node.remember(f"pre-skew observation {i}")
    await node.chaos.inject("clock_skew", duration_s=0.2, ms=120_000)
    for i in range(6):
        await node.remember(f"post-skew observation {i}")

    result = await node.sync.reconcile(trigger="skew")
    assert result.get("state") == "CONVERGED"
    assert node.sync.queue.depth == 0
    assert node.clock.max_observed_skew_ms >= 0


@pytest.mark.asyncio
async def test_repeated_restarts_are_idempotent(node, settings):
    from aegis.node import EdgeNode

    for i in range(10):
        await node.remember(f"observation {i}")
    node.store.archive()
    node.close()
    counts = []
    for _ in range(3):
        reopened = EdgeNode(settings)
        try:
            reopened.store.recover()
            counts.append(len(reopened.store.points))
        finally:
            reopened.close()
    assert len(set(counts)) == 1                 # replay converges to one state
    assert counts[0] >= 10


# -- the PII classifier must not be a denial-of-service vector ---------------

def test_classifier_cost_is_linear_in_input_size():
    """A stress run wedged this node for 16 minutes on one `remember()` call.

    The email pattern was unanchored, so a long run of word characters made it
    backtrack quadratically: measured at 4x the time for 2x the input, which
    extrapolates to 79 minutes for a 1 MB write on an unauthenticated ingest
    path. This asserts the shape of the cost curve, not a wall-clock number,
    so it stays meaningful on a slower machine than the one it was written on.
    """
    import time
    from aegis.inference.classifier import SensitivityClassifier

    classifier = SensitivityClassifier()
    timings = {}
    for size in (16_000, 64_000, 256_000):
        text = "a" * size
        start = time.perf_counter()
        classifier.classify(text)
        timings[size] = time.perf_counter() - start

    # 16x the input must not cost anything like 16^2 the time. Generous bound:
    # the quadratic version was ~256x here, the linear one is ~16x.
    ratio = timings[256_000] / max(timings[16_000], 1e-6)
    assert ratio < 48, f"cost grew {ratio:.0f}x for 16x the input — superlinear again"


def test_classifier_still_finds_pii_beyond_one_scan_window():
    """The fix must not have become a truncation.

    Bounding the scan is only acceptable while coverage is total: a secret
    past the first window is exactly the one somebody hid there.
    """
    from aegis.inference.classifier import SCAN_WINDOW, SensitivityClassifier

    classifier = SensitivityClassifier()
    buried = "x" * (SCAN_WINDOW * 3) + " write to jo.reyes@plant.io about it"
    result = classifier.classify(buried)
    assert "email" in result.signals
    assert result.sensitivity.value == "restricted"

    # ...including a match that straddles a window boundary.
    straddling = "y" * (SCAN_WINDOW - 8) + "jo.reyes@plant.io tail"
    assert "email" in classifier.classify(straddling).signals


def test_no_classifier_pattern_backtracks_catastrophically():
    """Guard the whole pattern set, not just the one that was found broken."""
    import time
    from aegis.inference.classifier import PATTERNS

    hostile = ["a" * 40_000, "1" * 40_000, "1 " * 20_000, "1-" * 20_000,
               "a." * 20_000, "a+-" * 13_000, "a" * 39_999 + "@"]
    for name, pattern, _ in PATTERNS:
        for text in hostile:
            start = time.perf_counter()
            pattern.search(text)
            elapsed = time.perf_counter() - start
            assert elapsed < 0.5, f"{name} took {elapsed:.2f}s on {len(text)} chars"
