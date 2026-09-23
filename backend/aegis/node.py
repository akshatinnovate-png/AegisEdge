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
from .core.scheduler import Lane, QoSScheduler
from .core.slo import Level, SLOManager
from .core.supervisor import Supervisor
from .core.tenancy import TenantRegistry
from .core.tracing import TRACER
from .inference.classifier import SensitivityClassifier
from .inference.embedder import Embedder
from .inference.governor import ThermalGovernor
from .inference.registry import ModelRegistry
from .inference.reranker import Reranker
from .inference.sparse import SparseEncoder
from .inference.triton import TritonClient
from .memory.consolidation import Consolidator
from .memory.graph import KnowledgeGraph
from .memory.repair import RepairCoordinator
from .memory.store import MemoryStore
from .policy.audit import AuditLog
from .policy.engine import PolicyEngine
from .policy.redaction import RedactionVault
from .renewal.migrator import DualSpaceMigrator
from .renewal.scheduler import RenewalScheduler
from .learning.adapter import RetrievalAdapter, TrainingExample
from .learning.federated import FederatedClient, FederatedCoordinator
from .learning.privacy import PrivacyBudget
from .retrieval.agent import ReasoningAgent
from .retrieval.conformal import ConformalPredictor
from .retrieval.diversity import MaximalMarginalRelevance
from .retrieval.late_interaction import LateInteractionIndex
from .retrieval.pipeline import RetrievalPipeline
from .retrieval.query_understanding import QueryUnderstanding
from .sync.crdt import Operation
from .sync.engine import SyncEngine
from .sync.gossip import GossipAgent, MeshLink
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
        self.scheduler = QoSScheduler(concurrency=self.settings.scheduler_concurrency)
        self.tracer = TRACER
        self.slo = SLOManager(self.bus, self.settings.slo_latency_ms, self.settings.slo_success)

        # -- governance
        self.policy = PolicyEngine(self.settings.policy_file)
        self.audit = AuditLog(Path(self.settings.data_dir) / "audit.log")
        self.vault = RedactionVault()
        self.tenants = TenantRegistry(self.audit)
        self.graph = KnowledgeGraph()

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
        self.understanding = QueryUnderstanding()
        self.conformal = ConformalPredictor(alpha=self.settings.conformal_alpha)
        self.diversity = MaximalMarginalRelevance()
        self.late_interaction = LateInteractionIndex(self.embedder)
        self.adapter = RetrievalAdapter(self.settings.memory.dim,
                                        rank=self.settings.learning.rank,
                                        alpha=self.settings.learning.alpha)
        self.adapter.load(Path(self.settings.data_dir) / "adapter.json")

        # -- memory
        self.store = MemoryStore(
            settings=self.settings, bus=self.bus, clock=self.clock, embedder=self.embedder,
            sparse=self.sparse, classifier=self.classifier, policy=self.policy,
            audit=self.audit, vault=self.vault, graph=self.graph, tenants=self.tenants,
        )
        self.segments = self.store.segments
        self.repair = RepairCoordinator(self)
        self.consolidator = Consolidator(self.store, self.settings.memory.consolidation_threshold)

        # -- sync
        self.transport = build_transport(self.settings.sync.cloud_url, self.settings.sync.bandwidth_bps)
        self.oracle = ConnectivityOracle(self.bus, self.transport, self.settings.sync.probe_interval_s)
        self.sync = SyncEngine(
            settings=self.settings, bus=self.bus, clock=self.clock, store=self.store,
            policy=self.policy, vault=self.vault, transport=self.transport, oracle=self.oracle,
        )

        # -- mesh (device-to-device, works with no cloud at all)
        self.mesh_link = MeshLink()
        self.mesh = GossipAgent(
            self.settings.node_id, self.mesh_link, self.bus,
            op_source=lambda: list(self.sync.oplog.ops),
            apply_op=self._apply_mesh_op,
            may_share=self._may_share_op,
        )

        # -- retrieval
        self.pipeline = RetrievalPipeline(
            store=self.store, sparse=self.sparse, reranker=self.reranker, triton=self.triton,
            oracle=self.oracle, bus=self.bus, half_life_days=self.settings.renewal.half_life_days,
            understanding=self.understanding, late_interaction=self.late_interaction,
            adapter=self.adapter if self.settings.learning.enabled else None,
            graph=self.graph, conformal=self.conformal, diversity=self.diversity, slo=self.slo,
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

        # -- learning
        self.federation = FederatedClient(
            self.settings.node_id, self.adapter,
            PrivacyBudget(epsilon_total=self.settings.learning.epsilon_total),
            epsilon_per_round=self.settings.learning.epsilon_per_round,
            differential_privacy=self.settings.learning.differential_privacy,
        )
        self.coordinator = FederatedCoordinator(self.adapter.parameters().size)
        self.feedback_buffer: list[TrainingExample] = []

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

        self.supervisor.register("scheduler", self.scheduler.run)
        self.supervisor.register("oracle", self.oracle.run)
        if self.settings.mesh_enabled:
            self.supervisor.register("mesh", self._mesh_loop)
        if self.settings.sync.enabled:
            self.supervisor.register("sync", self.sync.run)
        if self.settings.renewal.enabled:
            self.supervisor.register("renewal", self.renewal.run)
        self.supervisor.register("compactor", self._compaction_loop)
        self.supervisor.register("consolidator", self._consolidation_loop)
        self.supervisor.register("governor", self._governor_loop)
        self.supervisor.register("telemetry", self._telemetry_loop)
        self.supervisor.register("archiver", self._archive_loop)
        self.supervisor.register("scrubber", self._scrub_loop)
        self.supervisor.register("slo", self._slo_loop)
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
        self.scheduler.stop()
        await self.supervisor.stop_all()
        for index in getattr(self.store.store, "indexes", {}).values():
            index.storage.flush()
        self.store.wal.close()

    async def seed(self) -> int:
        """Seed through the same path a real write takes.

        Going straight to the store would skip vocabulary learning, the
        per-token index and mesh notification — so the node would boot with
        memories it cannot spell-correct against or gossip.
        """
        for collection, text in SEED_MEMORIES:
            await self.remember(text, collection=collection, source="seed")
        # one restricted memory, to prove the policy path end to end
        await self.remember(
            "Operator 4471 (jo.reyes@plant.io) acknowledged the alarm from console 2.",
            collection="episodic", source="seed",
        )
        return len(self.store.points)

    # -- background loops --------------------------------------------------

    async def _compaction_loop(self) -> None:
        while True:
            await asyncio.sleep(self.settings.memory.compaction_interval_s)
            try:
                await self.scheduler.submit("compact", self._compact_once, Lane.MAINTENANCE)
            except (RuntimeError, TimeoutError):
                continue          # shed under pressure: a query matters more

    async def _compact_once(self) -> dict[str, float]:
        return self.store.compact().as_dict()

    async def _mesh_loop(self) -> None:
        while True:
            await asyncio.sleep(self.settings.mesh_interval_s)
            if not self.mesh.peers:
                continue
            try:
                await self.scheduler.submit("mesh_round", lambda: self.mesh.round(2), Lane.SYNC)
            except (RuntimeError, TimeoutError):
                continue

    # -- mesh plumbing -----------------------------------------------------

    def _may_share_op(self, op: Operation) -> bool:
        """A peer is egress too: the same policy decides what may cross."""
        point = self.store.points.get(op.point_id)
        if point is None:
            return bool(op.body) and op.body.get("sensitivity") != "restricted"
        return self.policy.may_egress(point)

    async def _apply_mesh_op(self, op: Operation) -> None:
        await self.sync._materialize(op)
        self.pipeline.cache.invalidate()

    async def _consolidation_loop(self) -> None:
        while True:
            await asyncio.sleep(self.settings.memory.consolidation_interval_s)
            await self.consolidator.run("episodic")

    async def _governor_loop(self) -> None:
        while True:
            await asyncio.sleep(4.0)
            self.governor.sample()
            self.governor.enforce()

    async def _archive_loop(self) -> None:
        while True:
            await asyncio.sleep(self.settings.archive_interval_s)
            if not self.slo.allows("background_maintenance"):
                continue
            try:
                await self.scheduler.submit("archive", self._archive_once, Lane.MAINTENANCE)
            except (RuntimeError, TimeoutError):
                continue

    async def _archive_once(self) -> dict[str, Any] | None:
        return self.store.archive()

    async def _scrub_loop(self) -> None:
        """Bit rot is silent; the only way to find it is to read data nobody asked for."""
        while True:
            await asyncio.sleep(self.settings.scrub_interval_s)
            if not self.slo.allows("background_maintenance"):
                continue
            try:
                await self.scheduler.submit("scrub", lambda: self.repair.scrub(True),
                                            Lane.MAINTENANCE, budget_ms=30_000)
            except (RuntimeError, TimeoutError):
                continue

    async def _slo_loop(self) -> None:
        while True:
            await asyncio.sleep(self.settings.slo_interval_s)
            self.slo.evaluate(pressure=self.scheduler.pressure)

    async def _telemetry_loop(self) -> None:
        while True:
            await asyncio.sleep(self.settings.telemetry_interval_s)
            self.bus.publish("telemetry", "tick", memory=self.store.stats(),
                             node=self.state(), sync=self.sync.status())

    # -- ingest convenience -------------------------------------------------

    async def remember(self, text: str, collection: str = "episodic",
                       payload: dict[str, Any] | None = None, source: str | None = None,
                       tenant_id: str = "default") -> Any:
        point = await self.store.ingest(text, collection=collection, payload=payload,
                                        source=source, tenant_id=tenant_id)
        op = self.sync.record_local(point)
        self.understanding.observe(text)                  # vocabulary + co-occurrence
        self.late_interaction.index(point.id, text)       # per-token vectors
        if op is not None and self.settings.mesh_enabled:
            self.mesh.note_local(op)
        self.pipeline.cache.invalidate()
        return point

    # -- learning ----------------------------------------------------------

    async def feedback(self, query: str, chosen_id: str, rejected_id: str | None = None,
                       weight: float = 1.0) -> dict[str, Any]:
        """Teach the adapter from a real choice a person made."""
        chosen = self.store.points.get(chosen_id)
        if chosen is None or not chosen.dense:
            return {"accepted": False, "reason": "unknown or unembedded point"}
        rejected = self.store.points.get(rejected_id) if rejected_id else None
        query_vector = self.embedder.embed_sync([query])[0]
        example = TrainingExample(
            query=query_vector,
            positive=chosen.dense,
            negative=rejected.dense if rejected and rejected.dense else None,
            weight=weight,
        )
        self.feedback_buffer.append(example)
        self.audit.record("feedback", chosen_id, query=query, rejected=rejected_id)

        # A labelled choice is exactly the sample conformal calibration needs:
        # what the correct answer scored, against the best score on offer.
        try:
            probe = await self.pipeline.search(query, k=8, explain=False, understand=False)
            scores = {row["id"]: row["score"] for row in probe.results}
            if scores:
                top = max(scores.values())
                self.conformal.observe(scores.get(chosen_id, min(scores.values())), top)
                self.conformal.record_outcome(chosen_id in set(probe.confidence.get(
                    "prediction_set", list(scores))))
        except Exception:
            pass                      # calibration is best-effort; never fail a write on it
        loss = 0.0
        if len(self.feedback_buffer) >= self.settings.learning.batch:
            loss = self.adapter.learn(self.feedback_buffer)
            self.feedback_buffer.clear()
            self.adapter.save(Path(self.settings.data_dir) / "adapter.json")
            self.pipeline.cache.invalidate()
            self.bus.publish("learning", "adapter_updated", loss=round(loss, 5),
                             version=self.adapter.version,
                             message=(f"adapter updated · v{self.adapter.version} · "
                                      f"loss {loss:.4f}"))
        return {"accepted": True, "buffered": len(self.feedback_buffer),
                "loss": round(loss, 5), "adapter_version": self.adapter.version}

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
            "scheduler": self.scheduler.snapshot(),
            "slo": self.slo.snapshot(),
            "mesh_peers": self.mesh.snapshot()["alive_peers"],
            "adapter_version": self.adapter.version,
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
            "degradation": self.slo.level.name,
            "tenants": len(self.tenants.tenants),
            "graph_facts": len(self.graph.facts),
            "subsystems": self.supervisor.health(),
            "fallback_encoder": self.embedder.session.fallback,
        }
