"""Authentication and tenant resolution.

Auth is off by default because a single-tenant device on a private network
should not need a credential to answer its own operator. The moment
`AEGIS_REQUIRE_AUTH` is set, every route resolves a tenant from the
credential, and the tenant it resolves is the *only* one that request can
touch — a caller cannot name a tenant it did not authenticate as.
"""
from __future__ import annotations

from fastapi import Depends, Header, HTTPException, Request

from ..core.tenancy import ApiKey, Scope, TenantIsolationError
from ..node import EdgeNode
from .deps import get_node


class Principal:
    """Who is making this request, and what they may do."""

    def __init__(self, tenant_id: str, scopes: set[Scope], key: ApiKey | None = None,
                 anonymous: bool = False) -> None:
        self.tenant_id = tenant_id
        self.scopes = scopes
        self.key = key
        self.anonymous = anonymous

    def require(self, scope: Scope) -> None:
        if self.anonymous or Scope.ADMIN in self.scopes or scope in self.scopes:
            return
        raise HTTPException(status_code=403, detail=f"credential lacks scope '{scope.value}'")

    def as_dict(self) -> dict:
        return {"tenant_id": self.tenant_id, "anonymous": self.anonymous,
                "scopes": sorted(s.value for s in self.scopes),
                "key_id": self.key.key_id if self.key else None}


async def principal(
    request: Request,
    node: EdgeNode = Depends(get_node),
    authorization: str | None = Header(default=None),
    x_aegis_key: str | None = Header(default=None),
) -> Principal:
    secret = x_aegis_key
    if not secret and authorization and authorization.lower().startswith("bearer "):
        secret = authorization[7:].strip()

    if not node.settings.require_auth:
        # Open mode: still resolve a tenant if one was presented, so a
        # multi-tenant device behaves identically with and without enforcement.
        if secret:
            try:
                key = node.tenants.authenticate(secret)
                return Principal(key.tenant_id, key.scopes, key)
            except TenantIsolationError:
                pass
        return Principal("default", {Scope.ADMIN}, anonymous=True)

    if not secret:
        raise HTTPException(status_code=401, detail="credential required",
                            headers={"WWW-Authenticate": "Bearer"})
    try:
        key = node.tenants.authenticate(secret)
    except TenantIsolationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    return Principal(key.tenant_id, key.scopes, key)


def requires(scope: Scope):
    """Route dependency: `Depends(requires(Scope.WRITE))`."""
    async def dependency(who: Principal = Depends(principal)) -> Principal:
        who.require(scope)
        return who
    return dependency
