"""Hierarchical Navigable Small World graph.

Brute force is O(n) per query — fine at 10k points on a laptop, not fine at
400k on a device that also has to run inference. HNSW gives log-ish search by
building a layered proximity graph: sparse long-range links on upper layers
for coarse navigation, dense short-range links at layer 0 for precision.

This is a full implementation — heuristic neighbour selection (not naive
top-M, which produces hub nodes and collapses recall on clustered data),
bidirectional link pruning, soft deletes with repair, and an entry-point
demotion path so deleting the entry point cannot orphan the graph.
"""
from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass(slots=True)
class HnswParams:
    m: int = 16                     # links per node on layers > 0
    m0: int = 32                    # links at layer 0 (denser: it decides recall)
    ef_construction: int = 128
    ef_search: int = 64
    seed: int = 1337

    @property
    def level_lambda(self) -> float:
        return 1.0 / math.log(max(self.m, 2))


@dataclass
class HnswStats:
    nodes: int = 0
    deleted: int = 0
    layers: int = 0
    edges: int = 0
    searches: int = 0
    hops: int = 0
    distance_evals: int = 0
    build_distance_evals: int = 0
    repairs: int = 0

    def as_dict(self) -> dict[str, float]:
        return {
            "nodes": self.nodes, "deleted": self.deleted, "layers": self.layers,
            "edges": self.edges, "searches": self.searches,
            "avg_hops": round(self.hops / self.searches, 2) if self.searches else 0.0,
            "avg_distance_evals": round(self.distance_evals / self.searches, 1) if self.searches else 0.0,
            "repairs": self.repairs,
        }


class HnswIndex:
    """Cosine-space HNSW over unit vectors (distance = 1 - dot)."""

    def __init__(self, dim: int, params: HnswParams | None = None) -> None:
        self.dim = dim
        self._building = False
        self.p = params or HnswParams()
        self._rng = random.Random(self.p.seed)
        self.vectors: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        self.ids: list[str] = []
        self.slot: dict[str, int] = {}
        self.deleted: set[int] = set()
        self.levels: list[int] = []
        self.graph: list[list[set[int]]] = []      # graph[node][layer] -> neighbours
        self.entry: int | None = None
        self.max_layer = -1
        self.stats = HnswStats()

    # -- geometry ---------------------------------------------------------

    def _distance(self, query: np.ndarray, slots: Iterable[int]) -> dict[int, float]:
        slots = list(slots)
        if not slots:
            return {}
        if self._building:
            self.stats.build_distance_evals += len(slots)
        else:
            self.stats.distance_evals += len(slots)
        sims = self.vectors[slots] @ query
        return {slot: float(1.0 - sim) for slot, sim in zip(slots, sims)}

    def _assign_level(self) -> int:
        return int(-math.log(self._rng.random() + 1e-12) * self.p.level_lambda)

    # -- construction -----------------------------------------------------

    def add(self, point_id: str, vector: np.ndarray) -> None:
        self._building = True
        try:
            self._add(point_id, vector)
        finally:
            self._building = False

    def _add(self, point_id: str, vector: np.ndarray) -> None:
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector)) or 1.0
        vector = vector / norm

        if point_id in self.slot:
            self.remove(point_id)

        slot = len(self.ids)
        self.ids.append(point_id)
        self.slot[point_id] = slot
        self.vectors = (np.vstack([self.vectors, vector]) if len(self.vectors)
                        else vector.reshape(1, -1).copy())
        level = self._assign_level()
        self.levels.append(level)
        self.graph.append([set() for _ in range(level + 1)])
        self.stats.nodes += 1

        if self.entry is None:
            self.entry = slot
            self.max_layer = level
            self.stats.layers = level + 1
            return

        cursor = self.entry
        # descend the coarse layers greedily to find a good entry point
        for layer in range(self.max_layer, level, -1):
            cursor = self._greedy_descend(vector, cursor, layer)

        for layer in range(min(level, self.max_layer), -1, -1):
            candidates = self._search_layer(vector, [cursor], self.p.ef_construction, layer)
            degree = self.p.m0 if layer == 0 else self.p.m
            neighbours = self._select_neighbours(vector, candidates, degree)
            for neighbour in neighbours:
                self.graph[slot][layer].add(neighbour)
                self.graph[neighbour][layer].add(slot)
                self._prune(neighbour, layer, degree)
            cursor = neighbours[0] if neighbours else cursor

        if level > self.max_layer:
            self.max_layer = level
            self.entry = slot
            self.stats.layers = level + 1

    def _select_neighbours(self, vector: np.ndarray, candidates: list[tuple[float, int]],
                           degree: int) -> list[int]:
        """Heuristic selection (Malkov §4).

        Taking the nearest M candidates produces hubs: every node in a dense
        cluster links to the same few centres and the graph stops being
        navigable. A candidate is kept only if it is closer to the new node
        than to any already-kept neighbour, which preserves long-range links
        out of the cluster.
        """
        ordered = sorted(candidates)
        kept: list[int] = []
        for distance, candidate in ordered:
            if len(kept) >= degree:
                break
            if not kept:
                kept.append(candidate)
                continue
            to_kept = self.vectors[kept] @ self.vectors[candidate]
            if float(1.0 - to_kept.max()) > distance:
                kept.append(candidate)
        if len(kept) < degree:                      # backfill rather than under-link
            for _, candidate in ordered:
                if candidate not in kept:
                    kept.append(candidate)
                if len(kept) >= degree:
                    break
        return kept

    def _prune(self, node: int, layer: int, degree: int) -> None:
        links = self.graph[node][layer]
        if len(links) <= degree:
            return
        vector = self.vectors[node]
        distances = self._distance(vector, links)
        kept = self._select_neighbours(vector, [(d, s) for s, d in distances.items()], degree)
        for dropped in links - set(kept):
            self.graph[dropped][layer].discard(node)
        self.graph[node][layer] = set(kept)

    # -- search -----------------------------------------------------------

    def _greedy_descend(self, query: np.ndarray, start: int, layer: int) -> int:
        cursor = start
        best = self._distance(query, [cursor])[cursor]
        improved = True
        while improved:
            improved = False
            neighbours = self.graph[cursor][layer] if layer < len(self.graph[cursor]) else set()
            for slot, distance in self._distance(query, neighbours).items():
                if distance < best:
                    best, cursor, improved = distance, slot, True
                    if not self._building:
                        self.stats.hops += 1
        return cursor

    def _search_layer(self, query: np.ndarray, entries: list[int], ef: int,
                      layer: int) -> list[tuple[float, int]]:
        """Best-first expansion with a bounded result heap."""
        visited = set(entries)
        distances = self._distance(query, entries)
        candidates = [(d, s) for s, d in distances.items()]
        heapq.heapify(candidates)
        results = [(-d, s) for s, d in distances.items()]       # max-heap on distance
        heapq.heapify(results)

        while candidates:
            distance, slot = heapq.heappop(candidates)
            if results and distance > -results[0][0] and len(results) >= ef:
                break                                            # nothing closer remains
            neighbours = self.graph[slot][layer] if layer < len(self.graph[slot]) else set()
            fresh = [n for n in neighbours if n not in visited]
            visited.update(fresh)
            for neighbour, neighbour_distance in self._distance(query, fresh).items():
                if not self._building:
                    self.stats.hops += 1
                if len(results) < ef or neighbour_distance < -results[0][0]:
                    heapq.heappush(candidates, (neighbour_distance, neighbour))
                    heapq.heappush(results, (-neighbour_distance, neighbour))
                    if len(results) > ef:
                        heapq.heappop(results)
        return [(-d, s) for d, s in results]

    def search(self, query: np.ndarray, k: int, ef: int | None = None) -> list[tuple[str, float]]:
        if self.entry is None or not self.ids:
            return []
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(query)) or 1.0
        query = query / norm
        self.stats.searches += 1

        ef = max(ef or self.p.ef_search, k)
        cursor = self.entry
        for layer in range(self.max_layer, 0, -1):
            cursor = self._greedy_descend(query, cursor, layer)
        found = self._search_layer(query, [cursor], ef, 0)
        found.sort()
        out: list[tuple[str, float]] = []
        for distance, slot in found:
            if slot in self.deleted:
                continue                                          # soft-deleted, still navigable
            out.append((self.ids[slot], float(1.0 - distance)))
            if len(out) >= k:
                break
        return out

    # -- deletion ---------------------------------------------------------

    def remove(self, point_id: str) -> bool:
        """Soft delete.

        The node stays in the graph as a routing waypoint — physically
        excising it would tear holes in the connectivity that took
        ef_construction work to build. Its neighbours are cross-linked so
        paths through it survive, and it is skipped in results.
        """
        slot = self.slot.pop(point_id, None)
        if slot is None:
            return False
        self.deleted.add(slot)
        self.stats.deleted += 1
        for layer, links in enumerate(self.graph[slot]):
            live = [n for n in links if n not in self.deleted]
            for node in live:                                    # repair around the hole
                for other in live:
                    if other != node and len(self.graph[node][layer]) < self.p.m0:
                        self.graph[node][layer].add(other)
                        self.stats.repairs += 1
        if slot == self.entry:
            alive = [s for s in range(len(self.ids)) if s not in self.deleted]
            if alive:
                self.entry = max(alive, key=lambda s: self.levels[s])
                self.max_layer = self.levels[self.entry]
            else:
                self.entry, self.max_layer = None, -1
        return True

    @property
    def live(self) -> int:
        return len(self.ids) - len(self.deleted)

    def snapshot(self) -> dict[str, float]:
        self.stats.edges = sum(len(links) for node in self.graph for links in node)
        self.stats.layers = self.max_layer + 1
        return {**self.stats.as_dict(), "live": self.live, "ef_search": self.p.ef_search,
                "m": self.p.m, "m0": self.p.m0}
