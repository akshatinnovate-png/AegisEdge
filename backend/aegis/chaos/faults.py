"""Chaos harness.

Resilience claims are worth nothing undemonstrated. These faults are injected
from an admin endpoint so the failure modes the architecture claims to survive
can be triggered live, on stage, in front of judges.
"""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any

from ..core.bus import EventBus


@dataclass
class FaultRecord:
    fault: str
    params: dict[str, Any]
    at: float = field(default_factory=time.time)
    cleared_at: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"fault": self.fault, "params": self.params, "at": self.at,
                "cleared_at": self.cleared_at,
                "active": self.cleared_at is None}


class ChaosController:
    FAULTS = ("link_drop", "packet_loss", "latency_spike", "clock_skew",
              "disk_full", "thermal_spike", "corrupt_wal", "partition")

    def __init__(self, node, bus: EventBus) -> None:
        self.node = node
        self.bus = bus
        self.history: list[FaultRecord] = []

    async def inject(self, fault: str, duration_s: float = 20.0, **params: Any) -> dict[str, Any]:
        if fault not in self.FAULTS:
            raise ValueError(f"unknown fault '{fault}' — known: {', '.join(self.FAULTS)}")
        record = FaultRecord(fault=fault, params={"duration_s": duration_s, **params})
        self.history.append(record)
        self.bus.publish("alerts", "chaos_injected", level="error", fault=fault, **params,
                         message=f"chaos: <b>{fault}</b> injected for {duration_s:.0f}s")
        handler = getattr(self, f"_{fault}")
        await handler(duration_s, **params)
        asyncio.create_task(self._clear_after(record, duration_s))
        return record.as_dict()

    async def _clear_after(self, record: FaultRecord, duration_s: float) -> None:
        await asyncio.sleep(duration_s)
        clear = getattr(self, f"_clear_{record.fault}", None)
        if clear:
            await clear()
        record.cleared_at = time.time()
        self.bus.publish("alerts", "chaos_cleared", level="ok", fault=record.fault,
                         message=f"chaos: <b>{record.fault}</b> cleared")

    # -- faults ------------------------------------------------------------

    async def _link_drop(self, duration_s: float, **_: Any) -> None:
        self.node.oracle.forced_offline = True
        await self.node.oracle.probe_once()

    async def _clear_link_drop(self) -> None:
        self.node.oracle.forced_offline = False
        await self.node.oracle.probe_once()       # the restore callback fires from here

    async def _partition(self, duration_s: float, **_: Any) -> None:
        if hasattr(self.node.transport, "partition"):
            self.node.transport.partition(True)

    async def _clear_partition(self) -> None:
        if hasattr(self.node.transport, "partition"):
            self.node.transport.partition(False)

    async def _packet_loss(self, duration_s: float, rate: float = 0.35, **_: Any) -> None:
        if hasattr(self.node.transport, "loss"):
            self.node.transport.loss = rate

    async def _clear_packet_loss(self) -> None:
        if hasattr(self.node.transport, "loss"):
            self.node.transport.loss = 0.0

    async def _latency_spike(self, duration_s: float, ms: float = 450.0, **_: Any) -> None:
        if hasattr(self.node.transport, "latency_ms"):
            self.node.transport.latency_ms = ms

    async def _clear_latency_spike(self) -> None:
        if hasattr(self.node.transport, "latency_ms"):
            self.node.transport.latency_ms = 14.0

    async def _clock_skew(self, duration_s: float, ms: int = 9000, **_: Any) -> None:
        """Push the HLC forward: the CRDT layer has to keep converging anyway."""
        self.node.clock._wall += ms  # noqa: SLF001 - deliberate fault injection

    async def _thermal_spike(self, duration_s: float, temp_c: float = 94.0, **_: Any) -> None:
        self.node.governor.state.temp_c = temp_c
        self.node.governor.enforce()

    async def _disk_full(self, duration_s: float, **_: Any) -> None:
        self.node.store.wal.fsync = False
        self.bus.publish("alerts", "disk_pressure", level="error",
                         message="disk full — WAL degraded to buffered writes, ingest continues")

    async def _corrupt_wal(self, duration_s: float, bytes_: int = 64, **_: Any) -> None:
        """Append garbage to the WAL tail; recovery must truncate, not crash."""
        with open(self.node.store.wal.path, "a", encoding="utf-8") as fh:
            fh.write("deadbeef " + "".join(random.choice("0123456789abcdef") for _ in range(bytes_)) + "\n")

    def snapshot(self) -> dict[str, Any]:
        return {"available": list(self.FAULTS),
                "active": [r.as_dict() for r in self.history if r.cleared_at is None],
                "history": [r.as_dict() for r in self.history[-20:]]}
