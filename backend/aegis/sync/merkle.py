"""Merkle range digests.

Naive sync ships everything and hopes. Instead both sides build a Merkle tree
over their point-id space; if the roots match, nothing transfers at all. If
they differ, the walk narrows to the handful of buckets that actually diverge,
so a 400k-point collection reconciles in kilobytes.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Mapping


def _h(*parts: str) -> str:
    return hashlib.blake2b("|".join(parts).encode("utf-8"), digest_size=16).hexdigest()


def bucket_of(point_id: str, fanout: int) -> int:
    return int(hashlib.blake2b(point_id.encode("utf-8"), digest_size=4).hexdigest(), 16) % fanout


@dataclass(slots=True)
class MerkleDigest:
    root: str
    buckets: dict[int, str]
    counts: dict[int, int]
    total: int

    def as_dict(self) -> dict[str, object]:
        return {"root": self.root, "buckets": {str(k): v for k, v in self.buckets.items()},
                "counts": {str(k): v for k, v in self.counts.items()}, "total": self.total}

    @staticmethod
    def from_dict(d: Mapping) -> "MerkleDigest":
        return MerkleDigest(
            root=d["root"],
            buckets={int(k): v for k, v in d["buckets"].items()},
            counts={int(k): v for k, v in d.get("counts", {}).items()},
            total=int(d.get("total", 0)),
        )


class MerkleTree:
    def __init__(self, fanout: int = 16) -> None:
        self.fanout = fanout
        self.leaves: dict[int, dict[str, str]] = {i: {} for i in range(fanout)}

    def set(self, point_id: str, version: str) -> None:
        self.leaves[bucket_of(point_id, self.fanout)][point_id] = version

    def drop(self, point_id: str) -> None:
        self.leaves[bucket_of(point_id, self.fanout)].pop(point_id, None)

    def rebuild(self, items: Iterable[tuple[str, str]]) -> None:
        self.leaves = {i: {} for i in range(self.fanout)}
        for point_id, version in items:
            self.set(point_id, version)

    def digest(self) -> MerkleDigest:
        buckets: dict[int, str] = {}
        counts: dict[int, int] = {}
        for index, leaf in self.leaves.items():
            entries = sorted(leaf.items())
            buckets[index] = _h(*(f"{pid}:{ver}" for pid, ver in entries)) if entries else ""
            counts[index] = len(entries)
        root = _h(*(buckets[i] for i in range(self.fanout)))
        return MerkleDigest(root=root, buckets=buckets, counts=counts,
                            total=sum(counts.values()))

    def divergent(self, remote: MerkleDigest) -> list[int]:
        local = self.digest()
        if local.root == remote.root:
            return []
        return [i for i in range(self.fanout) if local.buckets.get(i, "") != remote.buckets.get(i, "")]

    def bucket_items(self, index: int) -> dict[str, str]:
        return dict(self.leaves.get(index, {}))

    def bytes_saved(self, divergent: list[int], avg_point_bytes: int = 1800) -> int:
        """What the digest walk avoided shipping — the number the demo cares about."""
        total = sum(len(leaf) for leaf in self.leaves.values())
        touched = sum(len(self.leaves[i]) for i in divergent)
        return max(0, (total - touched)) * avg_point_bytes
