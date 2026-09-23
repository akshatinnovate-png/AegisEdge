"""Data renewal: freshness, dual-space migration, shadow evaluation."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from ..node import EdgeNode
from .deps import get_node
from .models import MigrationRequest

router = APIRouter(prefix="/api/v1/renewal", tags=["renewal"])


@router.get("/status")
async def status(node: EdgeNode = Depends(get_node)) -> dict:
    return node.renewal.status()


@router.post("/sweep")
async def sweep(node: EdgeNode = Depends(get_node)) -> dict:
    return node.renewal.sweep().as_dict()


@router.post("/migrate")
async def migrate(body: MigrationRequest, node: EdgeNode = Depends(get_node)) -> dict:
    return node.migrator.begin(body.to_version)


@router.post("/migrate/step")
async def migrate_step(node: EdgeNode = Depends(get_node)) -> dict:
    return await node.migrator.step()
