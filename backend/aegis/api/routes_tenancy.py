"""Tenants, credentials and quotas."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..core.tenancy import Quota, Scope
from ..node import EdgeNode
from .deps import get_node
from .models import KeyRequest, TenantRequest
from .security import Principal, requires

router = APIRouter(prefix="/api/v1/tenants", tags=["tenancy"])


@router.get("")
async def list_tenants(node: EdgeNode = Depends(get_node),
                       who: Principal = Depends(requires(Scope.READ))) -> dict:
    snapshot = node.tenants.snapshot()
    if not who.anonymous and Scope.ADMIN not in who.scopes:
        snapshot["tenants"] = [t for t in snapshot["tenants"] if t["tenant_id"] == who.tenant_id]
    return snapshot


@router.post("")
async def create_tenant(body: TenantRequest, node: EdgeNode = Depends(get_node),
                        _: Principal = Depends(requires(Scope.ADMIN))) -> dict:
    quota = Quota(max_points=body.max_points, max_bytes=body.max_bytes,
                  max_qps=body.max_qps, max_ingest_per_minute=body.max_ingest_per_minute)
    return node.tenants.create(body.tenant_id, body.name, quota).as_dict()


@router.post("/{tenant_id}/keys")
async def issue_key(tenant_id: str, body: KeyRequest, node: EdgeNode = Depends(get_node),
                    _: Principal = Depends(requires(Scope.ADMIN))) -> dict:
    try:
        secret, record = node.tenants.issue_key(
            tenant_id, {Scope(s) for s in body.scopes}, body.label)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    # The secret is returned once and never stored in clear anywhere.
    return {"secret": secret, "key": record.as_dict(),
            "warning": "this secret is shown once; the node stores only its hash"}


@router.delete("/keys/{key_id}")
async def revoke_key(key_id: str, node: EdgeNode = Depends(get_node),
                     _: Principal = Depends(requires(Scope.ADMIN))) -> dict:
    if not node.tenants.revoke(key_id):
        raise HTTPException(status_code=404, detail="no such key")
    return {"revoked": key_id}


@router.post("/{tenant_id}/suspend")
async def suspend(tenant_id: str, node: EdgeNode = Depends(get_node),
                  _: Principal = Depends(requires(Scope.ADMIN))) -> dict:
    if not node.tenants.suspend(tenant_id):
        raise HTTPException(status_code=404, detail="no such tenant")
    return {"suspended": tenant_id}


@router.get("/whoami")
async def whoami(who: Principal = Depends(requires(Scope.READ))) -> dict:
    return who.as_dict()
