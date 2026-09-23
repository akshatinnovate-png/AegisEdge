"""EdgeNode — composition root.

Every subsystem is constructed here, in dependency order, and every background
loop runs under the supervisor. One object owns the node's lifecycle so the
API layer stays a thin translation of HTTP into calls on this.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from .chaos.faults import ChaosController
from .config import Settings, get_settings
from .core.bus import EventBus
from .core.clock import HybridClock
from .core.metrics import METRICS
from .core.supervisor import Supervisor
from .inference.classifier import SensitivityClassifier
from .inference.embedder import Embedder
from .inference.governor import ThermalGovernor
from .inference.registry import ModelRegistry
from .inference.reranker import Reranker
from .inference.sparse import SparseEncoder
from .inference.triton import TritonClient
from .memory.consolidation import Consolidator
from .memory.store import MemoryStore
from .policy.audit import AuditLog
from .policy.engine import PolicyEngine
from .policy.redaction import RedactionVault
from .renewal.migrator import DualSpaceMigrator
from .renewal.scheduler import RenewalScheduler
from .retrieval.agent import ReasoningAgent
from .retrieval.pipeline import RetrievalPipeline
from .sync.engine import SyncEngine
from .sync.oracle import ConnectivityOracle
from .sync.transport import build_transport

SEED_MEMORIES: list[tuple[str, str]] = [
    ("sensor", "Bay 3 conveyor vibration crossed 4.2 mm/s at 02:14; bearing signature matches the pre-failure cluster from March."),
    ("sensor", "Ambient temperature in the cell climbed to 61C during the night shift."),
    ("sensor", "Coolant pressure read 1.74 bar for 96 seconds before the interlock fired."),
    ("semantic", "Coolant pressure below 1.8 bar for over 90 seconds is treated as a hard stop condition on this cell."),
    ("semantic", "Restricted-class memories never leave this device under any sync policy."),
    ("semantic", "Bearing vibration above 4.0 mm/s is an early indicator of raceway spalling."),
    ("procedural", "Recovery: isolate the drive, purge the line, re-home the gantry, then release the interlock in that order."),
    ("procedural", "To clear a torque fault: cut servo power, rotate the spindle by hand, confirm free travel, then re-enable."),
    ("episodic", "Operator acknowledged the torque alarm and switched line 2 to manual feed for eleven minutes."),
    ("episodic", "Uplink dropped for 47 minutes during the night shift; operations queued locally and replayed on reconnect."),
    ("episodic", "Maintenance replaced the bay 3 bearing housing and logged the part number on the work order."),
]


class EdgeNode:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.started_at = time.time()
        self.ready = False

        # -- core
        self.bus = EventBus()
        self.clock = HybridClock(self.settings.node_id)
        self.supervisor = Supervisor(self.bus)

        # -- governance
        self.policy = PolicyEngine(self.settings.policy_file)
        self.audit = AuditLog(Path(self.settings.data_dir) / "audit.log")
        self.vault = RedactionVault()

        # -- inference
        self.registry = ModelRegistry(Path(self.settings.inference.model_dir))
        self.embedder = Embedder(
            self.registry, self.settings.inference.embedder, self.settings.memory.dim,
            self.settings.inference.batch_window_ms, self.settings.inference.max_batch,
            self.settings.inference.precision, self.settings.inference.model_dir,
        )
        self.sparse = SparseEncoder(self.settings.memory.sparse_dim)
        self.classifier = SensitivityClassifier()
        self.reranker = Reranker(self.settings.inference.reranker, self.settings.inference.model_dir)
        self.governor = ThermalGovernor(
            self.bus, self.embedder, self.settings.inference.thermal_ceiling_c,
            self.settings.inference.battery_floor_pct,
        )
        self.triton = TritonClient(url=None)

        # -- memory
        self.store = MemoryStore(
            settings=self.settings, bus=self.bus, clock=self.clock, embedder=self.embedder,
            sparse=self.sparse, classifier=self.classifier, policy=self.policy,
            audit=self.audit, vault=self.vault,
        )
        self.consolidator = Consolidator(self.store, self.settings.memory.consolidation_threshold)

        # -- sync
        self.transport = build_transport(self.settings.sync.cloud_url, self.settings.sync.bandwidth_bps)
        self.oracle = ConnectivityOracle(self.bus, self.transport, self.settings.sync.probe_interval_s)
        self.sync = SyncEngine(
            settings=self.settings, bus=self.bus, clock=self.clock, store=self.store,
            policy=self.policy, vault=self.vault, transport=self.transport, oracle=self.oracle,
        )

        # -- retrieval
        self.pipeline = RetrievalPipeline(
            store=self.store, sparse=self.sparse, reranker=self.reranker, triton=self.triton,
            oracle=self.oracle, bus=self.bus, half_life_days=self.settings.renewal.half_life_days,
        )
        self.agent = ReasoningAgent(self.pipeline, self.store, self.bus)

        # -- renewal
        self.migrator = DualSpaceMigrator(
            self.store, self.embedder, self.bus, Path(self.settings.data_dir),
            self.settings.renewal.batch,
        )
        self.renewal = RenewalScheduler(
            store=self.store, migrator=self.migrator, bus=self.bus,
            interval_s=self.settings.renewal.interval_s,
            half_life_days=self.settings.renewal.half_life_days,
        )

        # -- chaos
        self.chaos = ChaosController(self, self.bus)

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        recovery = self.store.recover()
        self.audit.record("boot", self.settings.node_id, wal=recovery)

        if self.settings.seed_demo and not self.store.points:
            await self.seed()

        for point in self.store.points.values():
            self.sync.tree.set(point.id, point.hlc or self.clock.now().pack())

        self.supervisor.register("oracle", self.oracle.run)
        if self.settings.sync.enabled:
            self.supervisor.register("sync", self.sync.run)
        if self.settings.renewal.enabled:
            self.supervisor.register("renewal", self.renewal.run)
        self.supervisor.register("compactor", self._compaction_loop)
        self.supervisor.register("consolidator", self._consolidation_loop)
        self.supervisor.register("governor", self._governor_loop)
        self.supervisor.register("telemetry", self._telemetry_loop)
        self.supervisor.start_all()

        self.ready = True
        self.bus.publish(
            "alerts", "node_ready", level="ok", node_id=self.settings.node_id,
            backend=self.store.store.backend, points=len(self.store.points),
            message=(f"node <b>{self.settings.node_id}</b> online · "
                     f"<b>{len(self.store.points)}</b> points · local memory authoritative"),
        )

    async def stop(self) -> None:
        self.ready = False
        await self.supervisor.stop_all()
        self.store.wal.close()

    async def seed(self) -> int:
        for collection, text in SEED_MEMORIES:
            point = await self.store.ingest(text, collection=collection, source="seed", origin="seed")
            self.sync.record_local(point)
        # one restricted memory, to prove the policy path end to end
        restricted = await self.store.ingest(
            "Operator 4471 (jo.reyes@plant.io) acknowledged the alarm from console 2.",
            collection="episodic", source="seed", origin="seed",
        )
        self.sync.record_local(restricted)
        return len(self.store.points)

    # -- background loops --------------------------------------------------

    async def _compaction_loop(self) -> None:
        while True:
            await asyncio.sleep(self.settings.memory.compaction_interval_s)
            self.store.compact()

    async def _consolidation_loop(self) -> None:
        while True:
            await asyncio.sleep(self.settings.memory.consolidation_interval_s)
            await self.consolidator.run("episodic")

    async def _governor_loop(self) -> None:
        while True:
            await asyncio.sleep(4.0)
            self.governor.sample()
            self.governor.enforce()

    async def _telemetry_loop(self) -> None:
        while True:
            await asyncio.sleep(self.settings.telemetry_interval_s)
            self.bus.publish("telemetry", "tick", memory=self.store.stats(),
                             node=self.state(), sync=self.sync.status())

    # -- ingest convenience -------------------------------------------------

    async def remember(self, text: str, collection: str = "episodic",
                       payload: dict[str, Any] | None = None, source: str | None = None) -> Any:
        point = await self.store.ingest(text, collection=collection, payload=payload, source=source)
        self.sync.record_local(point)
        self.pipeline.cache.invalidate()
        return point

    # -- reporting ----------------------------------------------------------

    def state(self) -> dict[str, Any]:
        embed_hist = METRICS.histograms.get("inference.embed_ms")
        return {
            "node_id": self.settings.node_id,
            "ready": self.ready,
            "uptime_s": round(time.time() - self.started_at, 1),
            "mode": "FUSED" if self.oracle.state.value != "offline" else "LOCAL",
            "link": self.oracle.snapshot(),
            "execution_provider": self.embedder.session.provider,
            "embedder": self.embedder.name,
            "model_version": self.embedder.version,
            "precision": self.embedder.entry.variant,
            "embed_p95_ms": round(embed_hist.quantile(0.95), 2) if embed_hist else None,
            "escalations": self.triton.escalations,
            "reconnect_ms": self.oracle.reconnect_ms,
            "governor": self.governor.snapshot(),
            "subsystems": self.supervisor.health(),
        }

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok" if self.ready and self.supervisor.healthy else "degraded",
            "node_id": self.settings.node_id,
            "uptime_s": round(time.time() - self.started_at, 1),
            "execution_provider": self.embedder.session.provider,
            "memory_backend": self.store.store.backend,
            "link": self.oracle.state.value,
            "points": len(self.store.points),
            "subsystems": self.supervisor.health(),
            "fallback_encoder": self.embedder.session.fallback,
        }
