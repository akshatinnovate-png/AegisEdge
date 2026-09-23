"""QoS scheduler.

Background work on an edge node is not optional — compaction, sync, renewal
and consolidation all have to run. But a query the operator is waiting on must
never queue behind a re-embedding batch.

So work is admitted into priority lanes with deadlines. Foreground work
preempts; background work yields and is shed under pressure rather than
allowed to pile up. Admission control rejects work the node cannot finish in
time instead of accepting everything and missing every deadline.
"""
from __future__ import annotations

import asyncio
import heapq
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Awaitable, Callable

from .metrics import METRICS


class Lane(IntEnum):
    INTERACTIVE = 0      # a person is waiting
    SYNC = 1             # the link is up and there is a window
    MAINTENANCE = 2      # compaction, consolidation
    RENEWAL = 3          # re-embedding: always last


@dataclass(order=True)
class _Job:
    sort_key: tuple[int, float]
    lane: Lane = field(compare=False)
    name: str = field(compare=False, default="")
    fn: Callable[[], Awaitable[Any]] = field(compare=False, default=None)  # type: ignore
    deadline: float = field(compare=False, default=0.0)
    enqueued_at: float = field(compare=False, default_factory=time.time)
    future: asyncio.Future | None = field(compare=False, default=None)


@dataclass
class LaneStats:
    admitted: int = 0
    rejected: int = 0
    completed: int = 0
    shed: int = 0
    deadline_misses: int = 0
    total_wait_ms: float = 0.0
    total_run_ms: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "admitted": self.admitted, "rejected": self.rejected, "completed": self.completed,
            "shed": self.shed, "deadline_misses": self.deadline_misses,
            "avg_wait_ms": round(self.total_wait_ms / self.completed, 2) if self.completed else 0.0,
            "avg_run_ms": round(self.total_run_ms / self.completed, 2) if self.completed else 0.0,
        }


class QoSScheduler:
    LANE_BUDGET_MS = {Lane.INTERACTIVE: 250.0, Lane.SYNC: 2_000.0,
                      Lane.MAINTENANCE: 5_000.0, Lane.RENEWAL: 10_000.0}
    QUEUE_LIMIT = {Lane.INTERACTIVE: 256, Lane.SYNC: 64, Lane.MAINTENANCE: 16, Lane.RENEWAL: 8}

    def __init__(self, concurrency: int = 2) -> None:
        self.concurrency = concurrency
        self._queue: list[_Job] = []
        self._depth: dict[Lane, int] = {lane: 0 for lane in Lane}
        self.stats: dict[Lane, LaneStats] = {lane: LaneStats() for lane in Lane}
        self._running = 0
        self._wake = asyncio.Event()
        self._stopped = False
        self.load = 0.0

    # -- admission --------------------------------------------------------

    def _admit(self, lane: Lane) -> bool:
        """Reject rather than accept work that will only miss its deadline."""
        if self._depth[lane] >= self.QUEUE_LIMIT[lane]:
            return False
        if lane >= Lane.MAINTENANCE and self._depth[Lane.INTERACTIVE] > 8:
            return False               # people are waiting; background work can come back later
        return True

    async def submit(self, name: str, fn: Callable[[], Awaitable[Any]], lane: Lane = Lane.INTERACTIVE,
                     budget_ms: float | None = None) -> Any:
        stats = self.stats[lane]
        if not self._admit(lane):
            stats.rejected += 1
            METRICS.incr(f"scheduler.rejected.{lane.name.lower()}")
            raise RuntimeError(f"admission control rejected '{name}' in lane {lane.name}")

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        deadline = time.time() + (budget_ms or self.LANE_BUDGET_MS[lane]) / 1000.0
        job = _Job(sort_key=(int(lane), deadline), lane=lane, name=name, fn=fn,
                   deadline=deadline, future=future)
        heapq.heappush(self._queue, job)
        self._depth[lane] += 1
        stats.admitted += 1
        self._wake.set()
        return await future

    # -- execution --------------------------------------------------------

    async def run(self) -> None:
        """Drain the queue, newest deadline first within the highest lane present."""
        while not self._stopped:
            if not self._queue or self._running >= self.concurrency:
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=0.25)
                except asyncio.TimeoutError:
                    pass
                continue
            job = heapq.heappop(self._queue)
            self._depth[job.lane] -= 1

            # shed stale background work rather than run it late
            if job.lane >= Lane.MAINTENANCE and time.time() > job.deadline:
                self.stats[job.lane].shed += 1
                if job.future and not job.future.done():
                    job.future.set_exception(TimeoutError(f"'{job.name}' shed past deadline"))
                continue

            self._running += 1
            asyncio.create_task(self._execute(job))

    async def _execute(self, job: _Job) -> None:
        stats = self.stats[job.lane]
        waited = (time.time() - job.enqueued_at) * 1000
        started = time.perf_counter()
        try:
            result = await job.fn()
            if job.future and not job.future.done():
                job.future.set_result(result)
        except Exception as exc:
            if job.future and not job.future.done():
                job.future.set_exception(exc)
        finally:
            elapsed = (time.perf_counter() - started) * 1000
            stats.completed += 1
            stats.total_wait_ms += waited
            stats.total_run_ms += elapsed
            if time.time() > job.deadline:
                stats.deadline_misses += 1
                METRICS.incr(f"scheduler.deadline_miss.{job.lane.name.lower()}")
            METRICS.observe(f"scheduler.run_ms.{job.lane.name.lower()}", elapsed)
            self._running -= 1
            self.load = self._running / self.concurrency
            self._wake.set()

    def stop(self) -> None:
        self._stopped = True
        self._wake.set()

    @property
    def pressure(self) -> float:
        """0..1 — how much the node is currently oversubscribed."""
        queued = sum(self._depth.values())
        return min(1.0, (self._running + queued) / max(self.concurrency * 4, 1))

    def snapshot(self) -> dict[str, Any]:
        return {
            "concurrency": self.concurrency, "running": self._running,
            "pressure": round(self.pressure, 3),
            "depth": {lane.name.lower(): self._depth[lane] for lane in Lane},
            "lanes": {lane.name.lower(): self.stats[lane].as_dict() for lane in Lane},
        }
