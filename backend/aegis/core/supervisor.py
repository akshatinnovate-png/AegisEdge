"""Subsystem supervisor: restart budgets and crash-loop detection.

An edge node has nobody to restart it. Every background loop runs under the
supervisor, which restarts it on failure — but only within a budget, so a
genuinely broken subsystem is quarantined and reported instead of spinning
the CPU forever.
"""
from __future__ import annotations

import asyncio
import time
import traceback
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from .bus import EventBus
from .metrics import METRICS


@dataclass
class TaskRecord:
    name: str
    factory: Callable[[], Awaitable[None]]
    restarts: int = 0
    budget: int = 5
    window_s: float = 60.0
    failures: list[float] = field(default_factory=list)
    state: str = "pending"
    last_error: str | None = None
    started_at: float = field(default_factory=time.time)
    task: asyncio.Task | None = None

    def crash_looping(self) -> bool:
        cutoff = time.time() - self.window_s
        self.failures = [f for f in self.failures if f >= cutoff]
        return len(self.failures) > self.budget


class Supervisor:
    def __init__(self, bus: EventBus) -> None:
        self.bus = bus
        self.records: dict[str, TaskRecord] = {}
        self._stopping = False

    def register(self, name: str, factory: Callable[[], Awaitable[None]], budget: int = 5) -> None:
        self.records[name] = TaskRecord(name=name, factory=factory, budget=budget)

    def start_all(self) -> None:
        for record in self.records.values():
            self._spawn(record)

    def _spawn(self, record: TaskRecord) -> None:
        record.state = "running"
        record.task = asyncio.create_task(self._run(record), name=f"aegis:{record.name}")

    async def _run(self, record: TaskRecord) -> None:
        while not self._stopping:
            try:
                await record.factory()
                record.state = "completed"
                return
            except asyncio.CancelledError:
                record.state = "cancelled"
                raise
            except Exception as exc:  # noqa: BLE001 - supervisor is the boundary
                record.failures.append(time.time())
                record.restarts += 1
                record.last_error = f"{type(exc).__name__}: {exc}"
                METRICS.incr(f"supervisor.restart.{record.name}")
                self.bus.publish(
                    "alerts", "subsystem_failed", level="error",
                    subsystem=record.name, error=record.last_error,
                    trace=traceback.format_exc(limit=3),
                    message=f"subsystem <b>{record.name}</b> failed: {record.last_error}",
                )
                if record.crash_looping():
                    record.state = "quarantined"
                    self.bus.publish(
                        "alerts", "subsystem_quarantined", level="error",
                        subsystem=record.name,
                        message=f"<b>{record.name}</b> crash-looping — quarantined",
                    )
                    return
                await asyncio.sleep(min(5.0, 0.25 * (2 ** min(record.restarts, 5))))

    async def stop_all(self) -> None:
        self._stopping = True
        tasks = [r.task for r in self.records.values() if r.task and not r.task.done()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def health(self) -> dict[str, dict[str, object]]:
        return {
            name: {
                "state": r.state,
                "restarts": r.restarts,
                "uptime_s": round(time.time() - r.started_at, 1),
                "last_error": r.last_error,
            }
            for name, r in self.records.items()
        }

    @property
    def healthy(self) -> bool:
        return all(r.state in {"running", "completed"} for r in self.records.values())
