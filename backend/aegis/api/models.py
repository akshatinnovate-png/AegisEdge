"""Request/response schemas — the contract the frontend is written against."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000)
    collection: str = "episodic"
    payload: dict[str, Any] = Field(default_factory=dict)
    source: str | None = None


class IngestResponse(BaseModel):
    id: str
    collection: str
    sensitivity: str
    sync_class: str
    policy_rule: str
    tier: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    k: int = Field(default=5, ge=1, le=50)
    collection: str = "*"
    mode: Literal["hybrid", "dense", "sparse"] = "hybrid"
    explain: bool = True
    allow_escalation: bool = True


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)


class SyncTriggerRequest(BaseModel):
    reason: str = "manual"


class ReviewRequest(BaseModel):
    point_id: str
    keep: Literal["local", "remote"]


class MigrationRequest(BaseModel):
    to_version: str = Field(min_length=1, max_length=120)


class ChaosRequest(BaseModel):
    duration_s: float = Field(default=20.0, ge=1.0, le=600.0)
    params: dict[str, Any] = Field(default_factory=dict)
