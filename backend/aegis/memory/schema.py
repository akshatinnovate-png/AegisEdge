"""The shape of a memory.

One record type carries everything the retrieval, policy, sync and renewal
subsystems each need to reason about a single remembered thing.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any

from ..core.ids import short_id


class Tier(str, Enum):
    HOT = "hot"        # mmap'd, full precision
    WARM = "warm"      # int8 scalar quantized
    COLD = "cold"      # binary quantized, payload on disk
    EVICTED = "evicted"


class Sensitivity(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    SENSITIVE = "sensitive"
    RESTRICTED = "restricted"

    @property
    def rank(self) -> int:
        return {"public": 0, "internal": 1, "sensitive": 2, "restricted": 3}[self.value]


class SyncClass(str, Enum):
    LOCAL_ONLY = "local_only"
    METADATA_ONLY = "sync_metadata_only"
    REDACTED = "sync_after_redaction"
    FULL = "sync_full"

    @property
    def restriction(self) -> int:
        """How tightly this class is held. Higher is more restrictive."""
        return {"sync_full": 0, "sync_after_redaction": 1,
                "sync_metadata_only": 2, "local_only": 3}[self.value]

    @classmethod
    def strictest(cls, *classes: "SyncClass") -> "SyncClass":
        """The most restrictive of several claims about one memory.

        A memory crossing devices carries its handling class with it, and the
        receiver must never relax it. Where the sender's class and the
        receiver's own policy disagree, the tighter one wins — that is the only
        direction in which being wrong is safe.
        """
        return max(classes, key=lambda c: c.restriction)


@dataclass(slots=True)
class MemoryPoint:
    id: str = field(default_factory=lambda: short_id("pt"))
    collection: str = "episodic"
    text: str = ""
    dense: list[float] = field(default_factory=list)
    sparse: dict[int, float] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)

    # provenance & lifecycle
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_access_at: float = field(default_factory=time.time)
    access_count: int = 0
    confidence: float = 1.0
    ttl_s: float | None = None
    pinned: bool = False
    stale: bool = False

    # placement & governance
    tier: Tier = Tier.HOT
    sensitivity: Sensitivity = Sensitivity.INTERNAL
    sync_class: SyncClass = SyncClass.FULL
    model_version: str = "bge-small-en-v1.5"

    # tenancy — isolation is structural, not a convention
    tenant_id: str = "default"

    # lineage
    device_id: str = ""
    hlc: str = ""
    superseded_by: str | None = None
    derived_from: list[str] = field(default_factory=list)
    source: str | None = None

    def age_s(self) -> float:
        return time.time() - self.created_at

    def expired(self) -> bool:
        return self.ttl_s is not None and self.age_s() > self.ttl_s and not self.pinned

    def touch(self) -> None:
        self.last_access_at = time.time()
        self.access_count += 1

    def summary(self, include_vectors: bool = False) -> dict[str, Any]:
        d = asdict(self)
        d["tier"] = self.tier.value
        d["sensitivity"] = self.sensitivity.value
        d["sync_class"] = self.sync_class.value
        if not include_vectors:
            d.pop("dense", None)
            d["sparse_terms"] = len(self.sparse)
            d.pop("sparse", None)
        return d
