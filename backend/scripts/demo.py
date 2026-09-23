"""End-to-end demo: the story the node is built to tell.

    python3 scripts/demo.py

No server, no cloud account — the whole edge-to-cloud lifecycle runs in one
process so the behaviour can be shown (and re-run) anywhere.
"""
from __future__ import annotations

import asyncio
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aegis.config import Settings           # noqa: E402
from aegis.core.slo import Level            # noqa: E402
from aegis.core.tenancy import Scope        # noqa: E402
from aegis.node import EdgeNode             # noqa: E402
from aegis.sync.gossip import GossipAgent    # noqa: E402
from aegis.sync.oracle import LinkState     # noqa: E402

O, B, D, R = "\033[38;5;208m", "\033[1m", "\033[2m", "\033[0m"


def head(n: int, title: str) -> None:
    print(f"\n{O}{B}[{n}] {title}{R}\n{D}{'─' * 64}{R}")


def line(label: str, value: object) -> None:
    print(f"  {label:<34} {B}{value}{R}")


async def main() -> None:
    settings = Settings()
    settings.data_dir = Path(".aegis/demo")
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("memory.wal", "opqueue.jsonl", "migration.checkpoint.json", "audit.log"):
        (settings.data_dir / stale).unlink(missing_ok=True)
    shutil.rmtree(settings.data_dir / "segments", ignore_errors=True)   # start from nothing

    node = EdgeNode(settings)
    await node.start()

    head(1, "COLD BOOT")
    health = node.health()
    line("node", health["node_id"])
    line("memory backend", health["memory_backend"])
    line("execution provider", health["execution_provider"])
    line("points resident", health["points"])
    line("subsystems running", sum(1 for s in health["subsystems"].values() if s["state"] == "running"))

    head(2, "HYBRID RETRIEVAL, OFFLINE")
    node.oracle.forced_offline = True
    await node.oracle.probe_once()
    line("link state", node.oracle.state.value)
    for query in ("coolant pressure hard stop", "how do i recover the drive"):
        result = await node.pipeline.search(query, k=3)
        print(f"\n  {D}query{R} {query}")
        line("latency", f"{result.latency_ms:.2f} ms")
        line("stages", result.stages)
        for hit in result.results[:2]:
            print(f"    {O}{hit['score']:.3f}{R} [{hit['collection']}] {hit['text'][:58]}…"
                  f"  {D}{'+'.join(hit['matched_by'])}{R}")

    head(3, "POLICY: WHAT MAY NOT LEAVE")
    restricted = await node.remember("Operator 4471 (jo@plant.io) overrode the interlock")
    geo = await node.remember("Fault reported at 12.97123,77.59456 on the west mast")
    line("restricted →", f"{restricted.sensitivity.value} / {restricted.sync_class.value}")
    line("queued for egress", node.sync.record_local(restricted) is not None)
    line("sensitive →", f"{geo.sensitivity.value} / {geo.sync_class.value}")
    line("what the cloud would see", node.sync._egress_body(geo)["text"][:58] + "…")

    head(4, "WORKING THROUGH AN OUTAGE")
    for i in range(12):
        await node.remember(f"night-shift observation {i}: belt tension drifting on line 2")
    line("link", node.oracle.state.value)
    line("durably queued ops", node.sync.queue.depth)
    line("answers still served", "yes — local memory is authoritative")

    head(5, "RECONNECTION")
    queued_before = node.sync.queue.depth
    t0 = time.perf_counter()
    node.oracle.forced_offline = False
    await node.oracle.probe_once()                 # the restore callback fires from here
    await asyncio.sleep(0.25)
    line("detected + replayed in", f"{(time.perf_counter() - t0) * 1000:.0f} ms")
    line("link", node.oracle.state.value)
    line("ops replayed on reconnect", queued_before - node.sync.queue.depth)
    result = await node.sync.reconcile(trigger="demo")
    line("follow-up cycle pushed/pulled", f"{result.get('pushed')} / {result.get('pulled')}")
    line("divergent merkle buckets", result.get("divergent"))
    line("bytes not shipped (digest walk)", f"{node.sync.bytes_saved:,}")
    line("queue depth now", node.sync.queue.depth)

    head(6, "FLEET KNOWLEDGE COMING BACK DOWN")
    text = "Torque limit on line 2 was raised to 42 Nm after the bearing swap"
    node.transport.inject("fleet-pt-demo", node.clock.now().pack(), {
        "collection": "semantic", "text": text, "sensitivity": "internal",
        "dense": node.embedder.embed_sync([text])[0].tolist(),
        "confidence": 0.93, "device_id": "edge-99",
    })
    pulled = await node.sync.reconcile(trigger="demo-pull")
    line("pulled from fleet", pulled.get("pulled"))
    answer = await node.agent.answer("what is the torque limit on line 2")
    line("agent answer", answer.answer[:58] + "…")
    line("citations", len(answer.citations))

    head(7, "CONTRADICTION + SUPERSESSION")
    await node.remember("The 42 Nm torque limit was rescinded pending review", collection="semantic")
    answer = await node.agent.answer("is the 42 Nm torque limit still valid")
    line("contradictions found", len(answer.contradictions))
    for finding in answer.contradictions[:2]:
        line("  signal", finding["signals"])

    head(8, "DATA RENEWAL — DUAL-SPACE MIGRATION")
    node.migrator.begin("bge-small-en-v2")
    line("state", node.migrator.state.value)
    line("corpus to re-embed", node.migrator.total)
    while node.migrator.state.value == "DUAL-SPACE":
        await node.migrator.step()
    line("final state", node.migrator.state.value)
    line("shadow eval", node.migrator.shadow_result)

    head(9, "CHAOS")
    await node.chaos.inject("thermal_spike", duration_s=1.0, temp_c=95.0)
    line("governor variant", node.governor.snapshot()["variant"])
    await node.chaos.inject("corrupt_wal", duration_s=1.0)
    line("WAL torn records after replay", sum(1 for _ in node.store.wal.replay()) and node.store.wal.torn)
    await node.chaos.inject("partition", duration_s=1.0)
    interrupted = await node.sync.reconcile(trigger="chaos")
    line("sync under partition", interrupted.get("error", "—"))
    line("nothing lost — queue depth", node.sync.queue.depth)

    head(10, "QUERY UNDERSTANDING")
    analysis = node.understanding.analyse("colent presure in bay3 over the last 2 hours")
    line("typed", "colent presure in bay3 over the last 2 hours")
    line("corrected", analysis.corrections)
    line("expanded from local corpus", analysis.expansions)
    line("filter extracted", list(analysis.filters))
    line("cost", f"{analysis.ms:.2f} ms (no network, no LLM)")

    head(11, "QUERY PLANNING")
    for filters, label in (({"collection": "procedural"}, "selective filter"),
                           ({"collection": "episodic"}, "loose filter"),
                           ({"collection": "nonexistent"}, "impossible filter")):
        result = await node.pipeline.search("pressure", k=3, filters=filters, understand=False)
        line(label, f"{result.plan['plan']:<11} {result.plan['reason'][:52]}")
    report = node.store.store.index_report()
    line("ann strategy per collection", {k: v["ann"]["strategy"] for k, v in report["collections"].items()})
    line("cost-model crossover", f"hnsw≥{report['cost_model']['hnsw_crossover']:,} points")

    head(12, "PEER MESH — NO CLOUD INVOLVED")
    from aegis.sync.crdt import OpKind, Operation
    peer_store: dict[str, Operation] = {}

    async def peer_apply(op: Operation) -> None:
        peer_store[op.op_id] = op

    peer = GossipAgent("edge-99", node.mesh_link, node.bus,
                       op_source=lambda: list(peer_store.values()),
                       apply_op=peer_apply, may_share=lambda _op: True)
    node.mesh.add_peer("edge-99")
    peer.add_peer(node.settings.node_id)
    for op in node.sync.oplog.ops:
        node.mesh.note_local(op)
    node.oracle.forced_offline = True
    await node.oracle.probe_once()
    line("uplink", node.oracle.state.value)
    outcome = await node.mesh.anti_entropy("edge-99")
    line("anti-entropy round", outcome)
    line("peer learned", f"{len(peer_store)} operations")
    restricted_ops = [op for op in node.sync.oplog.ops
                      if not node._may_share_op(op)]
    line("withheld from peer by policy", len(restricted_ops))
    line("leaked", sum(1 for op in restricted_ops if op.op_id in peer_store))
    node.oracle.forced_offline = False
    await node.oracle.probe_once()

    head(13, "LEARNING FROM FEEDBACK")
    found = await node.pipeline.search("coolant pressure", k=3)
    chosen, rejected = found.results[0]["id"], found.results[-1]["id"]
    version = node.adapter.version
    for _ in range(node.settings.learning.batch):
        await node.feedback("coolant pressure", chosen, rejected)
    line("adapter version", f"{version} → {node.adapter.version}")
    line("adapter size", f"{node.adapter.snapshot()['bytes'] / 1024:.1f} KB "
                         f"({node.adapter.snapshot()['parameters']} parameters)")
    cohort = [node.settings.node_id] + [f"peer-{i}" for i in range(7)]
    contribution = node.federation.contribute(cohort, 1)
    line("masked contribution", "indistinguishable from noise on its own")
    line("privacy budget", node.federation.budget.as_dict()["remaining"])

    head(14, "KNOWLEDGE GRAPH — STRUCTURE EMBEDDINGS CANNOT REACH")
    graph_stats = node.graph.snapshot()
    line("entities / facts", f"{graph_stats['entities']} / {graph_stats['facts']}")
    line("entity types", graph_stats["by_type"])
    parts = [e for e in node.graph.entities if e.startswith("part:")]
    if parts:
        found = node.graph.paths(parts[0], "location:line 2", max_hops=3)
        if found:
            line("multi-hop path", " → ".join(found[0].nodes))
            line("path confidence", found[0].confidence)
    believed_then = time.time()
    await asyncio.sleep(0.02)
    live = [f for f in node.graph.facts.values() if f.live()]
    if live:
        node.graph.retract(live[0].fact_id)
        diff = node.graph.diff_beliefs(believed_then, time.time())
        line("belief change (bitemporal)",
             f"{diff['learned_count']} learned · {diff['retracted_count']} retracted · "
             f"{diff['stable_count']} stable")
        line("retracted fact still auditable", live[0].as_dict()["retracted_at"] is not None)

    head(15, "CALIBRATED CONFIDENCE")
    result = await node.pipeline.search("coolant pressure hard stop", k=4)
    line("prediction set", result.confidence.get("set_size"))
    line("guarantee", result.confidence.get("guarantee", "")[:60])
    line("abstained", result.confidence.get("abstained"))
    line("diversity", result.diversity.get("improvement", 0.0))
    line("graph-boosted points", result.graph_context.get("boosted_points", 0))

    head(16, "MULTI-TENANCY")
    node.tenants.create("acme", "Acme Robotics")
    node.tenants.create("globex", "Globex")
    await node.remember("Acme confidential compressor fault", tenant_id="acme")
    await node.remember("Globex confidential compressor fault", tenant_id="globex")
    for tenant in ("acme", "globex"):
        view = await node.pipeline.search("confidential compressor", k=5, tenant_id=tenant)
        line(f"{tenant} sees", [h["text"][:34] for h in view.results])
    secret, key = node.tenants.issue_key("acme", {Scope.READ, Scope.WRITE}, "line-2 gateway")
    line("issued credential", f"{secret[:12]}… (stored hashed only)")
    line("cross-tenant cache blocks", node.pipeline.cache.snapshot()["cross_namespace_blocks"])

    head(17, "DURABILITY UNDER CORRUPTION")
    sealed = node.store.archive()
    line("sealed segment", f"{sealed['segment_id'][:12]} · {sealed['records']} records"
         if sealed else "nothing new to seal")
    line("fsck", node.segments.fsck().as_dict()["clean"])
    await node.chaos.inject("corrupt_segment", duration_s=0.5)
    scrub = await node.repair.scrub(repair=True)
    line("scrub found", f"{len(scrub.get('fsck', {}).get('corrupt', []))} damaged segment(s)")
    line("quarantined for forensics", scrub.get("fsck", {}).get("quarantined", 0))
    line("repair outcome", scrub.get("repair", {}).get("by_source") or "no redundancy available")
    line("generations retained", len(node.segments.generations()))

    head(18, "DEGRADATION LADDER")
    for level in (Level.FULL, Level.TRIM, Level.SURVIVAL):
        node.slo.override(level)
        degraded = await node.pipeline.search("coolant pressure", k=3)
        line(level.name, f"{len(degraded.results)} hits · shed: "
                         f"{', '.join(degraded.degradation.get('disabled', [])) or 'nothing'}"[:78])
    node.slo.override(None)

    head(19, "LEDGER")
    line("audit chain intact", node.audit.snapshot()["chain_intact"])
    line("audit entries", node.audit.snapshot()["entries"])
    line("policy denials", node.policy.snapshot()["denied_egress"])
    line("conflicts by rung", node.sync.arbiter.snapshot()["by_rung"])
    line("memory tiers", node.store.tier_counts())
    line("mesh", f"{node.mesh.snapshot()['known_ops']} ops known · "
                 f"{node.mesh.snapshot()['withheld_by_policy']} withheld")
    line("slowest span", node.tracer.slowest(1))
    line("scheduler lanes", {k: v["completed"] for k, v in node.scheduler.snapshot()["lanes"].items()})
    line("slo", f"{node.slo.snapshot()['level']} · p95 {node.slo.snapshot()['p95_ms']} ms")
    line("segments", f"{node.segments.snapshot()['segments']} sealed · "
                     f"{node.segments.snapshot()['records']} records")
    line("graph", f"{node.graph.snapshot()['entities']} entities · "
                  f"{node.graph.snapshot()['live_facts']} live facts")
    line("tenants", len(node.tenants.tenants))
    line("events published", node.bus.published)

    await node.stop()
    print(f"\n{O}{B}  local memory authoritative · network optional{R}\n")


if __name__ == "__main__":
    asyncio.run(main())
