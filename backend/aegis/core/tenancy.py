"""Multi-tenancy: namespaces, quotas, and isolation that is enforced.

One device often serves several parties — an OEM, the operator of the line,
and a maintenance contractor — and each one's memories must be invisible to
the others. Tenancy that is merely a payload field is not isolation: one
forgotten filter and a contractor reads the operator's incidents.

So isolation is structural. Every point carries a tenant, every query is
resolved against a tenant, and the *store* refuses cross-tenant reads rather
than trusting each call site to remember. Quotas are enforced on the write
path, because a tenant that can exhaust the device's memory has a denial of
service against everyone else on it.
"""
from __future__ import annotations

import hashlib
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .errors import AegisError
from .metrics import METRICS


class QuotaExceeded(AegisError):
    code = "quota_exceeded"


class TenantIsolationError(AegisError):
    code = "tenant_isolation"


class Scope(str, Enum):
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    SYNC = "sync"
    LEARN = "learn"


@dataclass
class Quota:
    max_points: int = 100_000
    max_bytes: int = 512 * 1024 * 1024
    max_qps: float = 50.0
    max_ingest_per_minute: int = 600
    burst: int = 25

    def as_dict(self) -> dict[str, Any]:
        return {"max_points": self.max_points, "max_bytes": self.max_bytes,
                "max_qps": self.max_qps, "max_ingest_per_minute": self.max_ingest_per_minute}


@dataclass
class Usage:
    points: int = 0
    bytes: int = 0
    queries: int = 0
    ingests: int = 0
    rejected: int = 0
    window_started: float = field(default_factory=time.time)
    window_queries: int = 0
    window_ingests: int = 0

    def roll(self, window_s: float = 60.0) -> None:
        if time.time() - self.window_started >= window_s:
            self.window_started = time.time()
            self.window_queries = 0
            self.window_ingests = 0

    def as_dict(self) -> dict[str, Any]:
        return {"points": self.points, "bytes": self.bytes, "queries": self.queries,
                "ingests": self.ingests, "rejected": self.rejected,
                "window_queries": self.window_queries, "window_ingests": self.window_ingests}


@dataclass
class Tenant:
    tenant_id: str
    name: str = ""
    quota: Quota = field(default_factory=Quota)
    usage: Usage = field(default_factory=Usage)
    key_id: str = ""                       # which at-rest key encrypts this tenant
    created_at: float = field(default_factory=time.time)
    active: bool = True
    collections: set[str] = field(default_factory=set)

    def namespace(self, collection: str) -> str:
        return f"{self.tenant_id}::{collection}"

    def as_dict(self) -> dict[str, Any]:
        return {"tenant_id": self.tenant_id, "name": self.name, "active": self.active,
                "quota": self.quota.as_dict(), "usage": self.usage.as_dict(),
                "key_id": self.key_id, "collections": sorted(self.collections),
                "headroom": {
                    "points": max(0, self.quota.max_points - self.usage.points),
                    "bytes": max(0, self.quota.max_bytes - self.usage.bytes),
                }}


@dataclass
class ApiKey:
    key_id: str
    tenant_id: str
    scopes: set[Scope]
    hashed: str
    label: str = ""
    created_at: float = field(default_factory=time.time)
    last_used: float = 0.0
    revoked: bool = False
    uses: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {"key_id": self.key_id, "tenant_id": self.tenant_id,
                "scopes": sorted(s.value for s in self.scopes), "label": self.label,
                "revoked": self.revoked, "uses": self.uses,
                "last_used": self.last_used or None}


DEFAULT_TENANT = "default"


class TenantRegistry:
    """Tenants, keys, quotas — the control plane for a shared device."""

    def __init__(self, audit=None) -> None:
        self.tenants: dict[str, Tenant] = {DEFAULT_TENANT: Tenant(DEFAULT_TENANT, "default")}
        self.keys: dict[str, ApiKey] = {}
        self.audit = audit
        self.denials = 0

    # -- tenants ----------------------------------------------------------

    def create(self, tenant_id: str, name: str = "", quota: Quota | None = None) -> Tenant:
        if tenant_id in self.tenants:
            return self.tenants[tenant_id]
        tenant = Tenant(tenant_id, name or tenant_id, quota or Quota(),
                        key_id=hashlib.blake2b(tenant_id.encode(), digest_size=8).hexdigest())
        self.tenants[tenant_id] = tenant
        if self.audit:
            self.audit.record("tenant_created", tenant_id, name=name)
        return tenant

    def get(self, tenant_id: str) -> Tenant:
        tenant = self.tenants.get(tenant_id)
        if tenant is None or not tenant.active:
            raise TenantIsolationError(f"unknown or inactive tenant '{tenant_id}'")
        return tenant

    def suspend(self, tenant_id: str) -> bool:
        tenant = self.tenants.get(tenant_id)
        if tenant is None:
            return False
        tenant.active = False
        if self.audit:
            self.audit.record("tenant_suspended", tenant_id)
        return True

    # -- keys -------------------------------------------------------------

    @staticmethod
    def _hash(secret: str) -> str:
        return hashlib.sha256(secret.encode("utf-8")).hexdigest()

    def issue_key(self, tenant_id: str, scopes: set[Scope], label: str = "") -> tuple[str, ApiKey]:
        """Returns (secret, record). The secret is shown once and never stored."""
        self.get(tenant_id)
        secret = "aeg_" + secrets.token_urlsafe(32)
        key_id = hashlib.blake2b(secret.encode(), digest_size=6).hexdigest()
        record = ApiKey(key_id=key_id, tenant_id=tenant_id, scopes=set(scopes),
                        hashed=self._hash(secret), label=label)
        self.keys[key_id] = record
        if self.audit:
            self.audit.record("key_issued", key_id, tenant=tenant_id,
                              scopes=sorted(s.value for s in scopes))
        return secret, record

    def authenticate(self, secret: str) -> ApiKey:
        key_id = hashlib.blake2b(secret.encode(), digest_size=6).hexdigest()
        record = self.keys.get(key_id)
        # constant-time comparison: a timing oracle on an API key is a real leak
        if record is None or record.revoked or not secrets.compare_digest(
                record.hashed, self._hash(secret)):
            self.denials += 1
            METRICS.incr("auth.denied")
            raise TenantIsolationError("invalid or revoked credential")
        record.uses += 1
        record.last_used = time.time()
        return record

    def revoke(self, key_id: str) -> bool:
        record = self.keys.get(key_id)
        if record is None:
            return False
        record.revoked = True
        if self.audit:
            self.audit.record("key_revoked", key_id, tenant=record.tenant_id)
        return True

    def authorize(self, key: ApiKey, scope: Scope) -> None:
        if Scope.ADMIN in key.scopes or scope in key.scopes:
            return
        self.denials += 1
        METRICS.incr("auth.forbidden")
        raise TenantIsolationError(f"credential lacks scope '{scope.value}'")

    # -- quotas -----------------------------------------------------------

    def check_write(self, tenant_id: str, points: int = 1, bytes_: int = 0) -> None:
        tenant = self.get(tenant_id)
        tenant.usage.roll()
        if tenant.usage.points + points > tenant.quota.max_points:
            tenant.usage.rejected += 1
            raise QuotaExceeded(
                f"tenant '{tenant_id}' at point quota "
                f"({tenant.usage.points}/{tenant.quota.max_points})")
        if tenant.usage.bytes + bytes_ > tenant.quota.max_bytes:
            tenant.usage.rejected += 1
            raise QuotaExceeded(f"tenant '{tenant_id}' at storage quota")
        if tenant.usage.window_ingests + points > tenant.quota.max_ingest_per_minute:
            tenant.usage.rejected += 1
            raise QuotaExceeded(f"tenant '{tenant_id}' exceeded ingest rate")
        tenant.usage.points += points
        tenant.usage.bytes += bytes_
        tenant.usage.ingests += points
        tenant.usage.window_ingests += points

    def check_read(self, tenant_id: str) -> None:
        tenant = self.get(tenant_id)
        tenant.usage.roll()
        elapsed = max(time.time() - tenant.usage.window_started, 1e-6)
        if tenant.usage.window_queries / elapsed > tenant.quota.max_qps + tenant.quota.burst:
            tenant.usage.rejected += 1
            raise QuotaExceeded(f"tenant '{tenant_id}' exceeded query rate")
        tenant.usage.queries += 1
        tenant.usage.window_queries += 1

    def release(self, tenant_id: str, points: int = 1, bytes_: int = 0) -> None:
        tenant = self.tenants.get(tenant_id)
        if tenant is None:
            return
        tenant.usage.points = max(0, tenant.usage.points - points)
        tenant.usage.bytes = max(0, tenant.usage.bytes - bytes_)

    def snapshot(self) -> dict[str, Any]:
        return {"tenants": [t.as_dict() for t in self.tenants.values()],
                "keys": len([k for k in self.keys.values() if not k.revoked]),
                "revoked_keys": len([k for k in self.keys.values() if k.revoked]),
                "denials": self.denials}
