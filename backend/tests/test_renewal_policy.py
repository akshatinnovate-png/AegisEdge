"""Renewal, policy and governance."""
from __future__ import annotations

import time

import pytest

from aegis.policy.audit import AuditLog
from aegis.policy.redaction import RedactionVault
from aegis.renewal.migrator import MigrationState


def test_audit_chain_detects_tampering():
    log = AuditLog()
    log.record("ingest", "p1")
    log.record("egress", "p1")
    assert log.verify() == (True, None)
    log.entries[0].detail["rule"] = "rewritten"
    intact, broken = log.verify()
    assert not intact and broken == 1


def test_redaction_is_reversible_only_locally():
    vault = RedactionVault()
    masked, kinds = vault.redact("reach jo@plant.io or +1 415 555 0134")
    assert "jo@plant.io" not in masked
    assert "EMAIL" in kinds
    assert vault.resolve(masked) == "reach jo@plant.io or +1 415 555 0134"


@pytest.mark.asyncio
async def test_policy_assigns_sync_class_by_collection(node):
    sensor = await node.remember("vibration 4.2 mm/s", collection="sensor")
    procedure = await node.remember("purge the line then re-home", collection="procedural")
    assert sensor.sync_class.value == "sync_metadata_only"
    assert procedure.sync_class.value == "sync_full" and procedure.pinned


@pytest.mark.asyncio
async def test_metadata_only_egress_carries_no_text(node):
    sensor = await node.remember("vibration 4.2 mm/s at bay 3", collection="sensor")
    body = node.sync._egress_body(sensor)
    assert body["metadata_only"] is True
    assert "text" not in body and "dense" not in body


@pytest.mark.asyncio
async def test_sensitive_egress_is_redacted(node):
    point = await node.remember("Site at 12.97123,77.59456 reported a fault")
    assert point.sensitivity.value == "sensitive"
    body = node.sync._egress_body(point)
    assert "12.97123" not in body["text"]
    assert body["redacted"]


@pytest.mark.asyncio
async def test_freshness_marks_old_memories_stale(node):
    point = await node.remember("a fact from long ago")
    point.updated_at = time.time() - 400 * 86400          # older than several half-lives
    report = node.renewal.sweep()
    assert report.marked_stale >= 1
    assert node.store.points[point.id].stale


@pytest.mark.asyncio
async def test_dual_space_migration_reembeds_and_promotes(node):
    for i in range(6):
        await node.remember(f"observation {i} for the migration corpus")
    node.migrator.begin("bge-small-en-v2")
    assert node.migrator.state is MigrationState.DUAL_SPACE
    assert node.migrator.status()["dual_space_active"]
    for _ in range(4):
        await node.migrator.step()
        if node.migrator.state is not MigrationState.DUAL_SPACE:
            break
    assert node.migrator.state in {MigrationState.COMPLETE, MigrationState.BLOCKED}
    assert all(p.model_version == "bge-small-en-v2" for p in node.store.points.values())
    assert node.migrator.shadow_result["golden_set"] > 0


@pytest.mark.asyncio
async def test_migration_checkpoint_survives_a_restart(node, settings):
    for i in range(4):
        await node.remember(f"checkpointed observation {i}")
    node.migrator.begin("bge-small-en-v3")
    node.migrator.batch = 2
    await node.migrator.step()
    done = len(node.migrator.checkpoint.done_ids)
    assert done > 0

    from aegis.renewal.migrator import DualSpaceMigrator
    from pathlib import Path

    reopened = DualSpaceMigrator(node.store, node.embedder, node.bus, Path(settings.data_dir))
    assert len(reopened.checkpoint.done_ids) == done      # resumes, does not restart
    assert reopened.state is MigrationState.DUAL_SPACE
