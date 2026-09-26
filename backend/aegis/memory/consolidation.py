"""Memory consolidation.

Without it, local memory grows linearly with uptime and the same observation
is remembered forty times. Near-duplicate episodic points are clustered and
distilled into one semantic point that keeps provenance links back to every
original, so nothing is silently lost.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..core.metrics import METRICS
from .schema import MemoryPoint, SyncClass, Tier


@dataclass(slots=True)
class ConsolidationReport:
    clusters: int = 0
    absorbed: int = 0
    created: int = 0
    duration_ms: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {"clusters": self.clusters, "absorbed": self.absorbed,
                "created": self.created, "duration_ms": round(self.duration_ms, 2)}


class Consolidator:
    def __init__(self, store, threshold: float = 0.94, min_cluster: int = 3) -> None:
        self.store = store
        self.threshold = threshold
        self.min_cluster = min_cluster
        self.runs = 0

    def _cluster(self, points: list[MemoryPoint]) -> list[list[MemoryPoint]]:
        """Greedy single-pass clustering — O(n·k), which is what an edge CPU can afford."""
        if len(points) < self.min_cluster:
            return []
        vectors = np.asarray([p.dense for p in points], dtype=np.float32)
        unassigned = set(range(len(points)))
        clusters: list[list[MemoryPoint]] = []
        while unassigned:
            seed = unassigned.pop()
            others = list(unassigned)
            sims = vectors[others] @ vectors[seed] if others else np.zeros(0)
            members = [seed]
            for idx, sim in zip(others, sims):
                if sim >= self.threshold:
                    members.append(idx)
            unassigned -= set(members)
            if len(members) >= self.min_cluster:
                clusters.append([points[i] for i in members])
        return clusters

    @staticmethod
    def _distil(cluster: list[MemoryPoint]) -> str:
        """Pick the medoid text and annotate the repetition count."""
        vectors = np.asarray([p.dense for p in cluster], dtype=np.float32)
        centroid = vectors.mean(axis=0)
        centroid /= np.linalg.norm(centroid) or 1.0
        medoid = cluster[int(np.argmax(vectors @ centroid))]
        span_h = (max(p.created_at for p in cluster) - min(p.created_at for p in cluster)) / 3600
        return f"{medoid.text} (observed {len(cluster)}x over {span_h:.1f}h)"

    async def run(self, collection: str = "episodic") -> ConsolidationReport:
        t0 = time.perf_counter()
        report = ConsolidationReport()
        candidates = [p for p in self.store.by_collection(collection)
                      if p.superseded_by is None and not p.pinned and p.has_dense]
        for cluster in self._cluster(candidates):
            summary = await self.store.ingest(
                self._distil(cluster), collection="semantic",
                payload={"consolidated_from": len(cluster), "origin_collection": collection},
                source="consolidation", confidence=min(0.99, 0.7 + 0.05 * len(cluster)),
                origin="consolidation",
            )
            summary.derived_from = [p.id for p in cluster]
            summary.tier = Tier.WARM
            summary.sync_class = max(
                (p.sync_class for p in cluster),
                key=lambda sc: ["local_only", "sync_metadata_only", "sync_after_redaction", "sync_full"].index(sc.value),
            ) if cluster else SyncClass.FULL
            for point in cluster:
                self.store.supersede(point.id, summary.id, reason="consolidated")
            report.clusters += 1
            report.created += 1
            report.absorbed += len(cluster)
        report.duration_ms = (time.perf_counter() - t0) * 1000
        self.runs += 1
        METRICS.incr("memory.consolidation_runs")
        if report.clusters:
            self.store.bus.publish(
                "memory", "consolidated", **report.as_dict(),
                message=(f"consolidated <b>{report.absorbed}</b> points into "
                         f"<b>{report.created}</b> semantic memories"),
            )
        return report
