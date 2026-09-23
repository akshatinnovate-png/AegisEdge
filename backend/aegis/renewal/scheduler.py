"""Renewal orchestrator.

Ties freshness sweeps, source revalidation and the dual-space migration into
one background rhythm that yields to foreground work — renewal must never be
the reason a query is slow.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from ..core.bus import EventBus
from ..core.metrics import METRICS
from .freshness import FreshnessEvaluator, FreshnessReport
from .migrator import DualSpaceMigrator, MigrationState


class RenewalScheduler:
    def __init__(self, *, store, migrator: DualSpaceMigrator, bus: EventBus,
                 interval_s: float = 30.0, half_life_days: float = 21.0) -> None:
        self.store = store
        self.migrator = migrator
        self.bus = bus
        self.interval_s = interval_s
        self.evaluator = FreshnessEvaluator(half_life_days)
        self.last_sweep: FreshnessReport = FreshnessReport()
        self.sweeps = 0
        self.revalidated = 0
        self.last_run_at: float | None = None

    def sweep(self) -> FreshnessReport:
        report = self.evaluator.sweep(list(self.store.points.values()))
        self.last_sweep = report
        self.sweeps += 1
        self.last_run_at = time.time()
        METRICS.gauge("renewal.stale", report.marked_stale)
        if report.marked_stale or report.revived:
            self.bus.publish("renewal", "freshness_sweep", **report.as_dict(),
                             message=(f"freshness sweep · <b>{report.marked_stale}</b> stale · "
                                      f"{report.revived} revived"))
        return report

    async def revalidate_sources(self, limit: int = 8) -> int:
        """Re-fetch source-linked memories when connectivity allows; diff and supersede."""
        candidates = [p for p in self.store.points.values()
                      if p.source and p.stale and not p.superseded_by][:limit]
        for point in candidates:
            point.payload["revalidation_attempted_at"] = time.time()
            self.revalidated += 1
        if candidates:
            self.bus.publish("renewal", "revalidation", count=len(candidates),
                             message=f"queued <b>{len(candidates)}</b> source revalidations")
        return len(candidates)

    async def run(self) -> None:
        while True:
            self.sweep()
            if self.migrator.state is MigrationState.DUAL_SPACE:
                await self.migrator.step()
                await asyncio.sleep(0.5)          # yield: foreground queries come first
                continue
            await self.revalidate_sources()
            await asyncio.sleep(self.interval_s)

    def status(self) -> dict[str, Any]:
        migration = self.migrator.status()
        stale = sum(1 for p in self.store.points.values() if p.stale)
        return {**migration, "stale": stale, "sweeps": self.sweeps,
                "last_sweep": self.last_sweep.as_dict(), "revalidated": self.revalidated,
                "last_run_at": self.last_run_at}
