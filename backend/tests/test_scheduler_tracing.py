"""QoS scheduling and span tracing."""
from __future__ import annotations

import asyncio
import time

import pytest

from aegis.core.scheduler import Lane, QoSScheduler
from aegis.core.tracing import Tracer


@pytest.mark.asyncio
async def test_interactive_work_runs_before_background_work():
    scheduler = QoSScheduler(concurrency=1)
    runner = asyncio.create_task(scheduler.run())
    order: list[str] = []

    async def work(tag: str):
        await asyncio.sleep(0.005)
        order.append(tag)

    await asyncio.gather(
        scheduler.submit("renewal", lambda: work("renewal"), Lane.RENEWAL),
        scheduler.submit("maint", lambda: work("maintenance"), Lane.MAINTENANCE),
        scheduler.submit("query", lambda: work("interactive"), Lane.INTERACTIVE),
    )
    assert order[0] == "interactive"
    scheduler.stop()
    runner.cancel()


@pytest.mark.asyncio
async def test_admission_control_rejects_background_work_under_load():
    scheduler = QoSScheduler(concurrency=1)
    scheduler._depth[Lane.INTERACTIVE] = 12            # a queue of waiting people
    with pytest.raises(RuntimeError, match="admission control"):
        await scheduler.submit("compact", lambda: asyncio.sleep(0), Lane.MAINTENANCE)
    assert scheduler.stats[Lane.MAINTENANCE].rejected == 1


@pytest.mark.asyncio
async def test_stale_background_work_is_shed_not_run_late():
    scheduler = QoSScheduler(concurrency=1)
    runner = asyncio.create_task(scheduler.run())
    ran = []

    async def work():
        ran.append(1)

    with pytest.raises(TimeoutError):
        await scheduler.submit("late", work, Lane.RENEWAL, budget_ms=0.001)
    await asyncio.sleep(0.02)
    assert not ran
    assert scheduler.stats[Lane.RENEWAL].shed == 1
    scheduler.stop()
    runner.cancel()


@pytest.mark.asyncio
async def test_exceptions_propagate_to_the_submitter():
    scheduler = QoSScheduler(concurrency=1)
    runner = asyncio.create_task(scheduler.run())

    async def boom():
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        await scheduler.submit("boom", boom, Lane.INTERACTIVE)
    scheduler.stop()
    runner.cancel()


def test_spans_nest_and_attribute_self_time():
    tracer = Tracer()
    with tracer.span("search") as root:
        with tracer.span("embed"):
            time.sleep(0.01)
        with tracer.span("recall"):
            with tracer.span("dense"):
                time.sleep(0.002)
    assert root.ms >= 12
    assert root.self_ms < root.ms
    assert root.hotspot()["name"] == "embed"
    assert len(root.flatten()) == 4


def test_span_records_failure_status():
    tracer = Tracer()
    try:
        with tracer.span("boom"):
            raise KeyError("x")
    except KeyError:
        pass
    assert tracer.completed[-1].status.startswith("error:KeyError")


@pytest.mark.asyncio
async def test_search_emits_a_usable_trace(node):
    await node.remember("Coolant pressure below 1.8 bar", collection="semantic")
    result = await node.pipeline.search("coolant pressure", k=3)
    names = {row["name"] for row in result.trace["children"]}
    assert {"embed", "recall", "rerank"} <= names | {c["name"] for c in result.trace["children"]}
    assert result.trace["ms"] > 0
