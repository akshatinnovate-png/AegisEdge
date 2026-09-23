"""Tenant isolation, quotas, credentials — the properties a breach would violate."""
from __future__ import annotations

import pytest

from aegis.core.tenancy import (Quota, QuotaExceeded, Scope, TenantIsolationError,
                                TenantRegistry)


@pytest.fixture
def registry() -> TenantRegistry:
    registry = TenantRegistry()
    registry.create("acme", "Acme Robotics", Quota(max_points=5, max_ingest_per_minute=5))
    registry.create("globex", "Globex")
    return registry


def test_credentials_are_stored_hashed(registry):
    secret, record = registry.issue_key("acme", {Scope.READ})
    assert secret not in record.hashed
    assert len(record.hashed) == 64
    assert registry.authenticate(secret).tenant_id == "acme"


def test_wrong_or_revoked_credentials_are_refused(registry):
    secret, record = registry.issue_key("acme", {Scope.READ})
    with pytest.raises(TenantIsolationError):
        registry.authenticate(secret + "tamper")
    registry.revoke(record.key_id)
    with pytest.raises(TenantIsolationError):
        registry.authenticate(secret)


def test_scopes_are_enforced_and_admin_implies_all(registry):
    _, reader = registry.issue_key("acme", {Scope.READ})
    _, admin = registry.issue_key("acme", {Scope.ADMIN})
    registry.authorize(reader, Scope.READ)
    with pytest.raises(TenantIsolationError):
        registry.authorize(reader, Scope.WRITE)
    registry.authorize(admin, Scope.WRITE)


def test_quotas_reject_before_work_is_done(registry):
    for _ in range(5):
        registry.check_write("acme")
    with pytest.raises(QuotaExceeded):
        registry.check_write("acme")
    assert registry.get("acme").usage.rejected == 1


def test_deleting_releases_quota(registry):
    registry.check_write("acme", points=3, bytes_=300)
    registry.release("acme", points=3, bytes_=300)
    assert registry.get("acme").usage.points == 0


def test_suspended_tenants_cannot_be_resolved(registry):
    registry.suspend("globex")
    with pytest.raises(TenantIsolationError):
        registry.get("globex")


def test_namespaces_do_not_collide(registry):
    assert registry.get("acme").namespace("episodic") != registry.get("globex").namespace("episodic")


# -- end-to-end isolation through the node ------------------------------------

@pytest.mark.asyncio
async def test_a_tenant_cannot_see_another_tenants_memories(node):
    node.tenants.create("acme")
    node.tenants.create("globex")
    await node.remember("Acme incident on line 9", tenant_id="acme")
    await node.remember("Globex incident on line 9", tenant_id="globex")

    acme = await node.pipeline.search("incident line 9", k=5, tenant_id="acme")
    globex = await node.pipeline.search("incident line 9", k=5, tenant_id="globex")
    assert [h["text"] for h in acme.results] == ["Acme incident on line 9"]
    assert [h["text"] for h in globex.results] == ["Globex incident on line 9"]


@pytest.mark.asyncio
async def test_the_semantic_cache_does_not_leak_across_tenants(node):
    """The bug this guards: same query text, same embedding, wrong tenant's rows."""
    node.tenants.create("acme")
    node.tenants.create("globex")
    await node.remember("Acme confidential incident", tenant_id="acme")
    await node.remember("Globex confidential incident", tenant_id="globex")

    first = await node.pipeline.search("confidential incident", k=3, tenant_id="acme")
    second = await node.pipeline.search("confidential incident", k=3, tenant_id="globex")
    assert all("Acme" not in hit["text"] for hit in second.results)
    assert node.pipeline.cache.snapshot()["cross_namespace_blocks"] >= 1
    assert first.results and second.results


@pytest.mark.asyncio
async def test_point_lookup_refuses_cross_tenant_reads(node):
    node.tenants.create("acme")
    point = await node.remember("Acme only", tenant_id="acme")
    assert node.store.get(point.id, "acme") is not None
    assert node.store.get(point.id, "default") is None
    assert node.store.get(point.id) is not None          # no tenant asserted: unchanged


@pytest.mark.asyncio
async def test_quota_blocks_ingest_before_embedding(node):
    from aegis.core.tenancy import Quota

    node.tenants.create("tiny", quota=Quota(max_points=2, max_ingest_per_minute=2))
    await node.remember("one", tenant_id="tiny")
    await node.remember("two", tenant_id="tiny")
    with pytest.raises(QuotaExceeded):
        await node.remember("three", tenant_id="tiny")
    assert node.store.stats()["by_tenant"].get("tiny") == 2
