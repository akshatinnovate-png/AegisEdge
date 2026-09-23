"""MemoryStore — the ingest path and the system of record.

Ordering here is the whole contract: classify, then apply policy, then embed,
then durably log, then index. Classification cannot be skipped by a caller,
and nothing becomes visible to search before it is recoverable from the WAL.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..core.bus import EventBus
from ..core.clock import HybridClock
from ..core.metrics import METRICS
from ..inference.classifier import SensitivityClassifier
from ..inference.embedder import Embedder
from ..inference.sparse import SparseEncoder
from ..policy.audit import AuditLog
from ..policy.engine import PolicyEngine
from ..policy.redaction import RedactionVault
from .schema import MemoryPoint, Sensitivity, SyncClass, Tier
from .tiering import CompactionReport, TieringPolicy
from .vectorstore import VectorStore, build_store
from .wal import WriteAheadLog


class MemoryStore:
    def __init__(
        self,
        *,
        settings,
        bus: EventBus,
        clock: HybridClock,
        embedder: Embedder,
        sparse: SparseEncoder,
        classifier: SensitivityClassifier,
        policy: PolicyEngine,
        audit: AuditLog,
        vault: RedactionVault,
    ) -> None:
        self.settings = settings
        self.bus = bus
        self.clock = clock
        self.embedder = embedder
        self.sparse = sparse
        self.classifier = classifier
        self.policy = policy
        self.audit = audit
        self.vault = vault

        data_dir = Path(settings.data_dir)
        self.wal = WriteAheadLog(data_dir / "memory.wal")
        self.store: VectorStore = build_store(
            settings.memory.dim, settings.memory.collections, str(data_dir / "qdrant")
        )
        self.points: dict[str, MemoryPoint] = {}
        self.tombstones: dict[str, float] = {}
        self.tiering = TieringPolicy(settings.memory.hot_capacity, settings.memory.warm_capacity)
        self.ingested = 0
        self.superseded = 0
        self.last_compaction = CompactionReport()

    # -- recovery ---------------------------------------------------------

    def recover(self) -> dict[str, Any]:
        """Replay the WAL. An unclean shutdown costs a few milliseconds, not data."""
        t0 = time.perf_counter()
        applied = 0
        for record in self.wal.replay():
            body = record.get("body", {})
            op = record.get("op")
            try:
                if op == "upsert":
                    self._apply_point(MemoryPoint(**self._decode(body)), log=False)
                    applied += 1
                elif op == "delete":
                    self._forget(body["id"], log=False)
                    applied += 1
                elif op == "supersede":
                    loser = self.points.get(body["loser"])
                    if loser:
                        loser.superseded_by = body["winner"]
                        applied += 1
            except Exception:
                continue        # a single unreadable record never blocks recovery
        stats = self.wal.stats() | {"applied": applied,
                                    "duration_ms": round((time.perf_counter() - t0) * 1000, 2)}
        self.bus.publish("memory", "wal_replayed", **stats,
                         message=f"WAL replayed · <b>{applied}</b> ops · {stats['torn']} torn")
        return stats

    @staticmethod
    def _decode(body: dict[str, Any]) -> dict[str, Any]:
        body = dict(body)
        body["tier"] = Tier(body.get("tier", "hot"))
        body["sensitivity"] = Sensitivity(body.get("sensitivity", "internal"))
        body["sync_class"] = SyncClass(body.get("sync_class", "sync_full"))
        body["sparse"] = {int(k): float(v) for k, v in (body.get("sparse") or {}).items()}
        return body

    # -- ingest -----------------------------------------------------------

    async def ingest(
        self,
        text: str,
        collection: str = "episodic",
        payload: dict[str, Any] | None = None,
        source: str | None = None,
        confidence: float = 1.0,
        origin: str = "local",
    ) -> MemoryPoint:
        with METRICS.timer("memory.ingest_ms"):
            point = MemoryPoint(
                collection=collection, text=text, payload=payload or {},
                source=source, confidence=confidence,
                device_id=self.settings.node_id, model_version=self.embedder.version,
            )

            classification = self.classifier.classify(text)          # 1. classify
            point.sensitivity = classification.sensitivity

            decision = self.policy.evaluate(point)                   # 2. govern
            point.sync_class = decision.sync_class
            point.ttl_s = decision.ttl_s
            point.pinned = decision.pin
            point.payload.setdefault("policy_rule", decision.rule)

            dense = await self.embedder.embed(text)                  # 3. embed
            point.dense = np.asarray(dense, dtype=np.float32).tolist()
            point.sparse = self.sparse.encode(text, fit=True)
            point.hlc = self.clock.now().pack()

            self._apply_point(point)                                 # 4. log + index
        self.ingested += 1
        METRICS.incr("memory.ingested")
        self.audit.record("ingest", point.id, collection=collection, rule=decision.rule,
                          sensitivity=point.sensitivity.value, origin=origin,
                          signals=classification.signals)
        self.bus.publish(
            "memory", "ingested", point_id=point.id, collection=collection,
            sensitivity=point.sensitivity.value, sync_class=point.sync_class.value,
            level="warn" if point.sensitivity is Sensitivity.RESTRICTED else "info",
            message=(f"ingested <b>{collection}</b> · {point.sensitivity.value}"
                     + (" · <b>local-only</b>" if point.sync_class is SyncClass.LOCAL_ONLY else "")),
        )
        return point

    def _apply_point(self, point: MemoryPoint, log: bool = True) -> None:
        if log:
            self.wal.append("upsert", point.summary(include_vectors=True))
        self.points[point.id] = point
        self.store.upsert(point)

    def _forget(self, point_id: str, log: bool = True) -> bool:
        if point_id not in self.points:
            return False
        if log:
            self.wal.append("delete", {"id": point_id})
        self.points.pop(point_id, None)
        self.store.delete(point_id)
        self.tombstones[point_id] = time.time()
        return True

    # -- mutation ---------------------------------------------------------

    def delete(self, point_id: str) -> bool:
        removed = self._forget(point_id)
        if removed:
            self.audit.record("delete", point_id)
            self.bus.publish("memory", "deleted", point_id=point_id,
                             message=f"tombstoned <b>{point_id}</b>")
        return removed

    def supersede(self, loser_id: str, winner_id: str, reason: str = "contradiction") -> bool:
        loser, winner = self.points.get(loser_id), self.points.get(winner_id)
        if not loser or not winner:
            return False
        loser.superseded_by = winner_id
        loser.confidence *= 0.5
        winner.derived_from = list({*winner.derived_from, loser_id})
        self.wal.append("supersede", {"loser": loser_id, "winner": winner_id, "reason": reason})
        self.superseded += 1
        self.audit.record("supersede", loser_id, winner=winner_id, reason=reason)
        self.bus.publish("memory", "superseded", level="warn", loser=loser_id, winner=winner_id,
                         message=f"<b>{loser_id}</b> superseded by <b>{winner_id}</b> ({reason})")
        return True

    def apply_remote(self, point: MemoryPoint) -> None:
        """Insert a point received from the cloud; never re-classified locally."""
        self._apply_point(point)

    # -- access -----------------------------------------------------------

    def get(self, point_id: str) -> MemoryPoint | None:
        point = self.points.get(point_id)
        if point:
            point.touch()
        return point

    def live_points(self) -> list[MemoryPoint]:
        return [p for p in self.points.values() if p.superseded_by is None]

    def by_collection(self, collection: str) -> Iterable[MemoryPoint]:
        return (p for p in self.points.values() if p.collection == collection)

    # -- maintenance ------------------------------------------------------

    def compact(self) -> CompactionReport:
        t0 = time.perf_counter()
        report = CompactionReport()
        points = list(self.points.values())
        plan = self.tiering.plan(points)
        for point in points:
            report.scanned += 1
            target = plan.get(point.id, point.tier)
            if target is Tier.EVICTED:
                self._forget(point.id)
                report.evicted += 1
                continue
            if target is point.tier:
                continue
            if self.store.move(point.id, target):
                order = {Tier.HOT: 0, Tier.WARM: 1, Tier.COLD: 2, Tier.EVICTED: 3}
                if order[target] < order[point.tier]:
                    report.promoted += 1
                else:
                    report.demoted += 1
                point.tier = target
        report.duration_ms = (time.perf_counter() - t0) * 1000
        self.last_compaction = report
        self.wal.checkpoint()
        if report.promoted or report.demoted or report.evicted:
            self.bus.publish("memory", "compacted", **report.as_dict(),
                             message=(f"compactor · <b>{report.promoted}</b>↑ "
                                      f"<b>{report.demoted}</b>↓ {report.evicted} evicted"))
        return report

    # -- reporting --------------------------------------------------------

    def tier_counts(self) -> dict[str, int]:
        counts = {t.value: 0 for t in (Tier.HOT, Tier.WARM, Tier.COLD)}
        for point in self.points.values():
            if point.tier.value in counts:
                counts[point.tier.value] += 1
        return counts

    def stats(self) -> dict[str, Any]:
        counts = self.tier_counts()
        by_sensitivity: dict[str, int] = {}
        by_collection: dict[str, int] = {}
        stale = 0
        for point in self.points.values():
            by_sensitivity[point.sensitivity.value] = by_sensitivity.get(point.sensitivity.value, 0) + 1
            by_collection[point.collection] = by_collection.get(point.collection, 0) + 1
            stale += int(point.stale)
        dim = self.settings.memory.dim
        resident_bytes = (counts["hot"] * dim * 4) + (counts["warm"] * (dim + 4)) + (counts["cold"] * dim // 8)
        return {
            **counts,
            "total": len(self.points),
            "collections": len(by_collection) or len(self.settings.memory.collections),
            "by_collection": by_collection,
            "by_sensitivity": by_sensitivity,
            "stale": stale,
            "superseded": self.superseded,
            "tombstones": len(self.tombstones),
            "backend": self.store.backend,
            "resident_bytes": resident_bytes,
            "wal": self.wal.stats(),
            "last_compaction": self.last_compaction.as_dict(),
        }
