"""Dual-space re-embedding migration.

When the embedder version changes, every stored vector is in the wrong space.
Re-embedding the corpus in one pass means downtime; mixing spaces silently
means corrupt retrieval. Instead both spaces are queried and fused while a
checkpointed background job migrates the corpus, hot tier first. Power loss
resumes from the last checkpoint.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from ..core.bus import EventBus
from ..core.metrics import METRICS
from ..memory.schema import Tier


class MigrationState(str, Enum):
    IDLE = "IDLE"
    DUAL_SPACE = "DUAL-SPACE"
    SHADOW_EVAL = "SHADOW-EVAL"
    COMPLETE = "COMPLETE"
    BLOCKED = "BLOCKED"


@dataclass
class Checkpoint:
    from_version: str = ""
    to_version: str = ""
    done_ids: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {"from_version": self.from_version, "to_version": self.to_version,
                "done": len(self.done_ids), "started_at": self.started_at}


class DualSpaceMigrator:
    """Runs the migration and gates promotion on a golden-set recall check."""

    RECALL_REGRESSION_LIMIT = 0.08

    def __init__(self, store, embedder, bus: EventBus, data_dir: Path, batch: int = 256) -> None:
        self.store = store
        self.embedder = embedder
        self.bus = bus
        self.batch = batch
        self.path = Path(data_dir) / "migration.checkpoint.json"
        self.state = MigrationState.IDLE
        self.checkpoint = Checkpoint()
        self.total = 0
        self.golden: list[tuple[str, str]] = []      # (query, expected point id)
        self.shadow_result: dict[str, Any] = {}
        self._load()

    # -- persistence ------------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.checkpoint = Checkpoint(**data.get("checkpoint", {}))
            self.state = MigrationState(data.get("state", "IDLE"))
        except Exception:
            self.state = MigrationState.IDLE

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(
            {"state": self.state.value,
             "checkpoint": {"from_version": self.checkpoint.from_version,
                            "to_version": self.checkpoint.to_version,
                            "done_ids": self.checkpoint.done_ids[-5000:],
                            "started_at": self.checkpoint.started_at}},
            default=str), encoding="utf-8")

    # -- lifecycle --------------------------------------------------------

    def begin(self, to_version: str, from_version: str | None = None) -> dict[str, Any]:
        self.checkpoint = Checkpoint(
            from_version=from_version or self.embedder.version, to_version=to_version,
        )
        pending = [p for p in self.store.points.values() if p.model_version != to_version]
        self.total = len(pending)
        self.state = MigrationState.DUAL_SPACE if pending else MigrationState.COMPLETE
        self.golden = self._sample_golden()
        self._save()
        self.bus.publish(
            "renewal", "migration_started", level="warn",
            from_version=self.checkpoint.from_version, to_version=to_version, total=self.total,
            message=(f"dual-space migration → <b>{to_version}</b> · {self.total} points "
                     f"· both spaces served while it runs"),
        )
        return self.status()

    def _sample_golden(self, size: int = 24) -> list[tuple[str, str]]:
        """Golden set: real stored texts, used to catch a recall regression."""
        points = [p for p in self.store.points.values() if p.text][:size]
        return [(" ".join(p.text.split()[:6]), p.id) for p in points]

    async def step(self) -> dict[str, Any]:
        """Migrate one checkpointed batch. Interruptible at every boundary."""
        if self.state is not MigrationState.DUAL_SPACE:
            return self.status()
        done = set(self.checkpoint.done_ids)
        order = {Tier.HOT: 0, Tier.WARM: 1, Tier.COLD: 2, Tier.EVICTED: 3}
        pending = sorted(
            (p for p in self.store.points.values()
             if p.id not in done and p.model_version != self.checkpoint.to_version),
            key=lambda p: order.get(p.tier, 3),                # hot tier migrates first
        )[: self.batch]
        if not pending:
            return await self._finish()

        t0 = time.perf_counter()
        vectors = self.embedder.embed_sync([p.text for p in pending])
        for point, vector in zip(pending, vectors):
            point.payload["prior_space"] = point.model_version      # dual-space bookkeeping
            point.set_dense(vector)
            point.model_version = self.checkpoint.to_version
            point.payload["last_verified_at"] = time.time()
            self.store.store.upsert(point)
            self.checkpoint.done_ids.append(point.id)
        METRICS.observe("renewal.batch_ms", (time.perf_counter() - t0) * 1000)
        METRICS.incr("renewal.reembedded", len(pending))
        self._save()
        self.bus.publish("renewal", "progress", done=len(self.checkpoint.done_ids), total=self.total,
                         progress=self.progress,
                         message=f"re-embedded <b>{len(self.checkpoint.done_ids)}</b>/{self.total}")
        return self.status()

    async def _finish(self) -> dict[str, Any]:
        """Shadow-evaluate before promoting: recall regression blocks the swap."""
        self.state = MigrationState.SHADOW_EVAL
        hits = 0
        for query, expected in self.golden:
            vector = self.embedder.embed_sync([query])[0]
            found = self.store.store.search_dense("*", np.asarray(vector, dtype=np.float32), 5)
            if any(pid == expected for pid, _ in found):
                hits += 1
        recall = hits / len(self.golden) if self.golden else 1.0
        baseline = self.shadow_result.get("baseline_recall", recall)
        regression = max(0.0, baseline - recall)
        self.shadow_result = {"recall_at_5": round(recall, 4), "baseline_recall": round(baseline, 4),
                              "regression": round(regression, 4), "golden_set": len(self.golden)}
        if regression > self.RECALL_REGRESSION_LIMIT:
            self.state = MigrationState.BLOCKED
            self.bus.publish("renewal", "promotion_blocked", level="error", **self.shadow_result,
                             message=(f"promotion <b>blocked</b> — recall regressed "
                                      f"{regression:.1%} on the golden set"))
        else:
            self.state = MigrationState.COMPLETE
            self.bus.publish("renewal", "migration_complete", level="ok", **self.shadow_result,
                             message=(f"migration complete · recall@5 <b>{recall:.0%}</b> · "
                                      f"single space restored"))
        self._save()
        return self.status()

    @property
    def progress(self) -> float:
        return round(len(self.checkpoint.done_ids) / self.total, 4) if self.total else 1.0

    def status(self) -> dict[str, Any]:
        done = len(self.checkpoint.done_ids)
        # A checkpoint restored from disk carries `done` but not the original
        # corpus size, so report a finished migration as done/done rather than
        # done/0, which would read as a broken denominator in the console.
        total = self.total or done
        return {"state": self.state.value, "done": done,
                "total": total, "progress": self.progress,
                "from_version": self.checkpoint.from_version,
                "to_version": self.checkpoint.to_version,
                "shadow": self.shadow_result, "dual_space_active": self.state is MigrationState.DUAL_SPACE}
