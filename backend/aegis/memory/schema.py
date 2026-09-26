"""The shape of a memory.

One record type carries everything the retrieval, policy, sync and renewal
subsystems each need to reason about a single remembered thing.
"""
from __future__ import annotations

import time
import numpy as np

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


# One shared empty array rather than a fresh allocation per point without a
# vector. It is read-only so a caller cannot make every such point share an
# accidental mutation.
EMPTY_DENSE = np.zeros(0, dtype=np.float32)
EMPTY_DENSE.flags.writeable = False


@dataclass(slots=True)
class MemoryPoint:
    id: str = field(default_factory=lambda: short_id("pt"))
    collection: str = "episodic"
    text: str = ""
    # float32, not a list of Python floats. Measured on this schema: a
    # 256-dimension vector costs 8,344 bytes as a list and 1,024 as an array,
    # because every element is a separate 24-byte object with a pointer to it.
    # At 18.6 KB for a whole point that one field was 45% of a device's
    # resident memory, and it is the field every point has.
    #
    # The trap this introduces is worth naming: `if point.dense` raises on an
    # array of more than one element. Use `point.has_dense`.
    dense: np.ndarray = field(default_factory=lambda: EMPTY_DENSE)
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

    def __post_init__(self) -> None:
        """Whatever a caller passes, a stored vector is float32.

        Normalising here rather than at each call site is deliberate. Points
        are constructed from the wire, from the WAL, from a peer's operation
        and from tests, and any one of those handing over a list would leave a
        point that costs eight times what it should and breaks `has_dense`
        besides. One door.
        """
        if not isinstance(self.dense, np.ndarray):
            self.set_dense(self.dense)

    @property
    def has_dense(self) -> bool:
        """`if point.dense` raises on an array. This is what to use instead."""
        return self.dense is not None and self.dense.size > 0

    def dense_list(self) -> list[float]:
        """For the wire and for JSON, where an array is not a value."""
        return self.dense.tolist() if self.has_dense else []

    def set_dense(self, vector: Any) -> None:
        """One place that decides what a stored vector is."""
        if vector is None:
            self.dense = EMPTY_DENSE
            return
        # A copy, always. `np.ascontiguousarray` returns the *same object*
        # when its input is already contiguous float32, which is precisely the
        # case the micro-batcher produces — so every point held a view into
        # the batch array it came from, pinning the whole buffer and exposed
        # to any later mutation of it. That is the opposite of what the comment
        # here used to claim.
        self.dense = np.array(vector, dtype=np.float32, copy=True, order="C")

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
        d["dense"] = self.dense_list()          # asdict leaves the array as-is
        if not include_vectors:
            d.pop("dense", None)
            d["sparse_terms"] = len(self.sparse)
            d.pop("sparse", None)
        return d
