"""Self-healing memory.

Detecting corruption is table stakes; a product that only detects it hands the
operator a broken device and a log line. A fleet has redundancy by
construction — the same memories exist on peers and, for anything policy
allowed to sync, in the cloud. So corruption becomes a *repair* rather than a
loss.

The ordering matters. Peers first: they are reachable when the uplink is not,
and repairing from a peer costs no cloud egress. Cloud second. Whatever
neither can supply is reported as permanently lost, with its identifiers, so
the loss is auditable instead of silent.

Local-only memories have no redundancy by design — that is the privacy
guarantee working as specified, and the report says so rather than implying
the data might come back.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..core.bus import EventBus
from ..core.metrics import METRICS


class RepairSource(str, Enum):
    PEER = "peer"
    CLOUD = "cloud"
    UNRECOVERABLE = "unrecoverable"
    LOCAL_ONLY = "local_only_by_policy"


@dataclass
class RepairReport:
    started_at: float = field(default_factory=time.time)
    damaged: list[str] = field(default_factory=list)
    still_resident: list[str] = field(default_factory=list)      # durable copy lost, memory intact
    recovered: dict[str, str] = field(default_factory=dict)      # point_id -> source
    unrecoverable: list[str] = field(default_factory=list)
    local_only: list[str] = field(default_factory=list)
    peers_asked: list[str] = field(default_factory=list)
    duration_ms: float = 0.0

    @property
    def complete(self) -> bool:
        return not self.unrecoverable

    def as_dict(self) -> dict[str, Any]:
        by_source: dict[str, int] = {}
        for source in self.recovered.values():
            by_source[source] = by_source.get(source, 0) + 1
        return {
            "damaged": len(self.damaged), "still_resident": len(self.still_resident),
            "recovered": len(self.recovered),
            "by_source": by_source, "unrecoverable": self.unrecoverable,
            "local_only_unrecoverable": self.local_only,
            "peers_asked": self.peers_asked, "complete": self.complete,
            "duration_ms": round(self.duration_ms, 2),
        }


class RepairCoordinator:
    def __init__(self, node) -> None:
        self.node = node
        self.bus: EventBus = node.bus
        self.repairs = 0
        self.points_recovered = 0
        self.last: RepairReport | None = None

    def assess(self, fsck_report) -> tuple[list[str], list[str]]:
        """Split the damage into what is *lost* and what merely needs re-sealing.

        A corrupt segment is not automatically lost data. Most of those
        memories are still resident in RAM — their durable copy is gone, not
        the memory itself, and the fix is to re-archive rather than to beg a
        peer for something we already hold. Counting those as "recovered from
        a peer" would be a flattering lie about what the repair did.
        """
        condemned = {row["segment_id"] for row in fsck_report.corrupt} | set(fsck_report.missing)
        if not condemned:
            return [], []
        affected: set[str] = set()
        for info in self.node.segments.manifest.segments:
            if info.segment_id not in condemned:
                continue
            try:
                from .segments import SegmentReader
                for record in SegmentReader(info.path).read(strict=False):
                    point_id = (record.get("body") or {}).get("id") or record.get("id")
                    if point_id:
                        affected.add(point_id)
            except Exception:
                continue
        lost = sorted(pid for pid in affected if pid not in self.node.store.points)
        needs_rearchive = sorted(pid for pid in affected if pid in self.node.store.points)
        return lost, needs_rearchive

    def damaged_point_ids(self, fsck_report) -> list[str]:
        """Only the memories that are genuinely gone from this node."""
        return self.assess(fsck_report)[0]

    async def repair(self, point_ids: list[str]) -> RepairReport:
        """Recover the named points from peers, then cloud."""
        started = time.perf_counter()
        report = RepairReport(damaged=list(point_ids))
        # Only points we do not already hold are candidates for recovery.
        outstanding = {pid for pid in point_ids if pid not in self.node.store.points}
        report.recovered.update({pid: "already_resident" for pid in point_ids
                                 if pid in self.node.store.points})
        self.repairs += 1

        # 1. peers — reachable when the uplink is not, and free
        for peer_id in list(self.node.mesh.peers):
            if not outstanding:
                break
            report.peers_asked.append(peer_id)
            try:
                await self.node.mesh.anti_entropy(peer_id)
            except Exception:
                continue
            healed = {pid for pid in outstanding if pid in self.node.store.points}
            for point_id in healed:
                report.recovered[point_id] = RepairSource.PEER.value
            outstanding -= healed

        # 2. cloud — only for points policy ever allowed to leave
        if outstanding and self.node.oracle.state.value != "offline":
            try:
                await self.node.sync.reconcile(trigger="repair")
                healed = {pid for pid in outstanding if pid in self.node.store.points}
                for point_id in healed:
                    report.recovered[point_id] = RepairSource.CLOUD.value
                outstanding -= healed
            except Exception:
                pass

        # 3. what is left is genuinely gone; say which, and why
        for point_id in sorted(outstanding):
            point = self.node.store.points.get(point_id)
            if point is not None and point.sync_class.value == "local_only":
                report.local_only.append(point_id)
            report.unrecoverable.append(point_id)

        report.duration_ms = (time.perf_counter() - started) * 1000
        self.points_recovered += len(report.recovered)
        self.last = report
        METRICS.incr("repair.runs")
        METRICS.incr("repair.points_recovered", len(report.recovered))
        self.bus.publish(
            "alerts", "repair", level="ok" if report.complete else "error",
            **report.as_dict(),
            message=(f"repair · recovered <b>{len(report.recovered)}</b>/{len(point_ids)}"
                     + ("" if report.complete
                        else f" · <b>{len(report.unrecoverable)}</b> unrecoverable")),
        )
        return report

    async def scrub(self, repair: bool = True) -> dict[str, Any]:
        """Background scrub: verify every segment, heal what is damaged.

        Bit rot is silent by definition — it is found by reading data nobody
        asked for. A node that only verifies on read discovers the damage the
        day it matters most.
        """
        fsck = self.node.segments.fsck(repair=False)
        if fsck.clean:
            return {"scrubbed": fsck.checked, "clean": True}

        lost, resident = self.assess(fsck)
        self.bus.publish(
            "alerts", "corruption_detected", level="error",
            segments=len(fsck.corrupt), lost=len(lost), still_resident=len(resident),
            message=(f"scrub found <b>{len(fsck.corrupt)}</b> damaged segment(s) · "
                     f"{len(lost)} memories lost · {len(resident)} still in memory"),
        )
        # Before quarantining, note the LSN range those segments covered: it is
        # about to stop being durable and must be re-sealed from the WAL.
        condemned = {row["segment_id"] for row in fsck.corrupt} | set(fsck.missing)
        lowest = min((info.min_lsn for info in self.node.segments.manifest.segments
                      if info.segment_id in condemned and info.min_lsn),
                     default=0)
        repaired = self.node.segments.fsck(repair=True)     # quarantine, then heal
        if lowest:
            self.node.store.rewind_archive(lowest)
        report = await self.repair(lost) if repair else RepairReport(damaged=lost)
        report.still_resident = resident

        # Memories we still hold simply need a fresh durable copy.
        resealed = self.node.store.archive() if resident and repair else None
        return {"scrubbed": fsck.checked, "clean": False, "fsck": repaired.as_dict(),
                "repair": report.as_dict(), "resealed": resealed}

    def snapshot(self) -> dict[str, Any]:
        return {"repairs": self.repairs, "points_recovered": self.points_recovered,
                "last": self.last.as_dict() if self.last else None}
