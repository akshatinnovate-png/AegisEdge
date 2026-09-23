"""Core primitives: clock ordering, breaker, backoff, bus backpressure."""
from __future__ import annotations

import asyncio

from aegis.core.bus import EventBus
from aegis.core.circuit import BreakerState, CircuitBreaker
from aegis.core.clock import HLC, HybridClock
from aegis.core.ids import ulid
from aegis.core.ratelimit import TokenBucket


def test_hlc_is_monotonic_and_roundtrips():
    clock = HybridClock("A")
    a, b = clock.now(), clock.now()
    assert b.dominates(a)
    assert HLC.parse(a.pack()) == a


def test_hlc_never_moves_backwards_on_skewed_peer():
    clock = HybridClock("A")
    local = clock.now()
    merged = clock.observe(HLC(local.wall_ms - 60_000, 0, "B"))
    assert merged.wall_ms >= local.wall_ms
    assert clock.max_observed_skew_ms > 0


def test_ulids_sort_by_creation():
    ids = [ulid() for _ in range(64)]
    assert ids == sorted(ids)


def test_breaker_opens_then_half_opens():
    breaker = CircuitBreaker("t", failure_threshold=2, reset_after_s=0.0)
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state is BreakerState.OPEN
    assert breaker.allows()
    assert breaker.state is BreakerState.HALF_OPEN
    breaker.record_success()
    assert breaker.state is BreakerState.CLOSED


def test_bus_drops_oldest_for_slow_subscriber():
    bus = EventBus()
    sub = bus.subscribe(maxsize=4)
    for i in range(10):
        bus.publish("telemetry", "tick", i=i)
    assert sub.queue.qsize() == 4
    assert sub.dropped == 6
    newest = [sub.queue.get_nowait().payload["i"] for _ in range(4)]
    assert newest == [6, 7, 8, 9]          # the newest survive, not the oldest


def test_token_bucket_throttles():
    bucket = TokenBucket(rate=100, capacity=10)
    assert bucket.try_take(10)
    assert not bucket.try_take(5)
    assert asyncio.run(bucket.take(5)) > 0
