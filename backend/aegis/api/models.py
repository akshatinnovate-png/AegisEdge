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
    filters: dict[str, Any] | None = Field(
        default=None,
        description='Payload filter, e.g. {"collection": "sensor", "ts": {"gte": 1700000000}}',
    )
    understand: bool = True


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)


class SyncTriggerRequest(BaseModel):
    reason: str = "manual"


class ReviewRequest(BaseModel):
    point_id: str
    keep: Literal["local", "remote"]


class MigrationRequest(BaseModel):
    to_version: str = Field(min_length=1, max_length=120)


class FeedbackRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    chosen_id: str
    rejected_id: str | None = None
    weight: float = Field(default=1.0, ge=0.0, le=10.0)


class FederatedRoundRequest(BaseModel):
    cohort: list[str] | None = None
    simulate_peers: int = Field(default=7, ge=0, le=512)


class PeerRequest(BaseModel):
    node_id: str = Field(min_length=1, max_length=64)
    endpoint: str = ""


class TenantRequest(BaseModel):
    tenant_id: str = Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_.-]+$")
    name: str = ""
    max_points: int = Field(default=100_000, ge=1)
    max_bytes: int = Field(default=512 * 1024 * 1024, ge=1024)
    max_qps: float = Field(default=50.0, gt=0)
    max_ingest_per_minute: int = Field(default=600, ge=1)


class KeyRequest(BaseModel):
    scopes: list[Literal["read", "write", "admin", "sync", "learn"]] = ["read"]
    label: str = ""


class PathRequest(BaseModel):
    source: str
    target: str
    max_hops: int = Field(default=3, ge=1, le=4)
    valid_time: float | None = None
    as_of: float | None = None


class RestoreRequest(BaseModel):
    generation: int = Field(ge=0)


class DegradationRequest(BaseModel):
    level: str | None = None


class ChaosRequest(BaseModel):
    duration_s: float = Field(default=20.0, ge=1.0, le=600.0)
    params: dict[str, Any] = Field(default_factory=dict)
