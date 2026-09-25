"""Hybrid logical clock.

Wall time alone cannot order events across devices whose clocks drift; a pure
Lamport counter loses human-readable time. The HLC keeps both, which is what
the CRDT layer needs to converge without a coordinator.
"""
from __future__ import annotations

import threading
import time

from . import determinism
from dataclasses import dataclass


@dataclass(frozen=True, slots=True, order=True)
class HLC:
    wall_ms: int
    counter: int
    node_id: str = ""

    def pack(self) -> str:
        return f"{self.wall_ms:013d}.{self.counter:05d}.{self.node_id}"

    @staticmethod
    def parse(raw: str) -> "HLC":
        wall, counter, node = raw.split(".", 2)
        return HLC(int(wall), int(counter), node)

    def dominates(self, other: "HLC") -> bool:
        """Strict happens-after, with node id as the deterministic tiebreak."""
        return (self.wall_ms, self.counter, self.node_id) > (
            other.wall_ms, other.counter, other.node_id,
        )


class HybridClock:
    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self._lock = threading.Lock()
        self._wall = 0
        self._counter = 0
        self.max_observed_skew_ms = 0

    def now(self) -> HLC:
        with self._lock:
            wall = int(determinism.now() * 1000)
            if wall > self._wall:
                self._wall, self._counter = wall, 0
            else:
                self._counter += 1
            return HLC(self._wall, self._counter, self.node_id)

    def observe(self, remote: HLC) -> HLC:
        """Merge a peer's timestamp; local time can never move backwards."""
        with self._lock:
            wall = int(determinism.now() * 1000)
            self.max_observed_skew_ms = max(
                self.max_observed_skew_ms, abs(remote.wall_ms - wall)
            )
            new_wall = max(self._wall, remote.wall_ms, wall)
            if new_wall == self._wall == remote.wall_ms:
                self._counter = max(self._counter, remote.counter) + 1
            elif new_wall == self._wall:
                self._counter += 1
            elif new_wall == remote.wall_ms:
                self._counter = remote.counter + 1
            else:
                self._counter = 0
            self._wall = new_wall
            return HLC(self._wall, self._counter, self.node_id)
