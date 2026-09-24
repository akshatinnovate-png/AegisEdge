"""Energy accounting: the number that decides whether an edge device is viable.

Latency is the figure every system reports and the wrong one to optimise a
battery-powered device against. A node that answers in 4 ms while holding four
cores at 100% is worse, on a robot or a handset, than one that answers in 12 ms
on a single core. What matters is joules per answer, and how many answers fit
in the charge the device is carrying.

Almost nothing measures this, so almost nothing can trade against it.

Three sources, tried in order, and the reading always says which one produced
it. The distinction is not pedantry: two of these are measurements and one is a
model, and a model presented as a measurement is the kind of number that
survives right up until somebody checks it.

1. **RAPL** (`/sys/class/powercap/intel-rapl*/energy_uj`) — the CPU package's
   own energy counter, in microjoules. This is a measurement: the silicon is
   telling you what it spent. Available on most Intel and recent AMD parts
   under Linux, and on nothing else.
2. **Battery discharge** — the difference in the pack's reported energy over a
   window. Also a measurement, and a coarse one: it covers the whole device
   rather than the process, and it only works while unplugged.
3. **CPU time x a stated coefficient** — a *model*. It multiplies CPU-seconds
   by an assumed watts-per-core, which is a guess about silicon this code has
   never seen. It is the fallback because some number is better than none, and
   it is labelled `modelled` everywhere it appears so nobody mistakes it for
   the first two.

The coefficient is configuration rather than a constant buried here, because
the right value for a Xeon in a rack and an A78 in a handset differ by an
order of magnitude and only the deployer knows which they have.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

RAPL_ROOT = Path("/sys/class/powercap")
BATTERY_ROOT = Path("/sys/class/power_supply")


class Source(str, Enum):
    RAPL = "rapl"                 # measured, per-package, microjoule counter
    BATTERY = "battery"           # measured, whole-device, coarse
    MODEL = "modelled"            # assumed watts per busy core
    NONE = "unavailable"

    @property
    def measured(self) -> bool:
        return self in (Source.RAPL, Source.BATTERY)


@dataclass(slots=True)
class Reading:
    """One energy observation, and the honesty about where it came from."""

    joules: float
    cpu_seconds: float
    wall_seconds: float
    source: Source
    ops: int = 1

    @property
    def joules_per_op(self) -> float:
        return self.joules / max(self.ops, 1)

    @property
    def watts(self) -> float:
        return self.joules / self.wall_seconds if self.wall_seconds > 0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"joules": round(self.joules, 6), "joules_per_op": round(self.joules_per_op, 6),
                "cpu_seconds": round(self.cpu_seconds, 4),
                "wall_seconds": round(self.wall_seconds, 4),
                "watts": round(self.watts, 3), "ops": self.ops,
                "source": self.source.value, "measured": self.source.measured}


@dataclass(slots=True)
class Account:
    """Running totals for one kind of operation."""

    kind: str
    ops: int = 0
    joules: float = 0.0
    cpu_seconds: float = 0.0
    wall_seconds: float = 0.0
    samples: list[float] = field(default_factory=list)      # joules per op

    def add(self, reading: Reading) -> None:
        self.ops += reading.ops
        self.joules += reading.joules
        self.cpu_seconds += reading.cpu_seconds
        self.wall_seconds += reading.wall_seconds
        self.samples.append(reading.joules_per_op)
        if len(self.samples) > 512:
            del self.samples[: len(self.samples) - 512]

    def as_dict(self) -> dict[str, Any]:
        per_op = self.joules / self.ops if self.ops else 0.0
        ordered = sorted(self.samples)
        return {
            "kind": self.kind, "ops": self.ops,
            "joules_total": round(self.joules, 4),
            "joules_per_op": round(per_op, 6),
            "millijoules_per_op": round(per_op * 1000, 4),
            "p50_joules_per_op": round(ordered[len(ordered) // 2], 6) if ordered else 0.0,
            "cpu_seconds": round(self.cpu_seconds, 3),
            "cpu_ms_per_op": round(self.cpu_seconds / self.ops * 1000, 3) if self.ops else 0.0,
        }


def _rapl_domains() -> list[Path]:
    if not RAPL_ROOT.exists():
        return []
    found = []
    for entry in sorted(RAPL_ROOT.glob("intel-rapl:*")):
        counter = entry / "energy_uj"
        try:
            counter.read_text()
            found.append(counter)
        except Exception:
            continue                       # present but not readable: not a source
    return found


def _battery_energy_wh() -> tuple[float | None, float | None]:
    """(energy now, energy full) in watt-hours, or (None, None)."""
    for battery in sorted(BATTERY_ROOT.glob("BAT*")):
        try:
            def read(name: str) -> float | None:
                path = battery / name
                return float(path.read_text().strip()) if path.exists() else None
            now, full = read("energy_now"), read("energy_full")
            if now is not None and full:
                return now / 1e6, full / 1e6            # microwatt-hours
            charge_now, charge_full = read("charge_now"), read("charge_full")
            volts = read("voltage_now")
            if charge_now is not None and charge_full and volts:
                return (charge_now * volts / 1e12, charge_full * volts / 1e12)
        except Exception:
            continue
    return None, None


class EnergyMeter:
    """Measure where the device allows it; model, and say so, where it does not."""

    def __init__(self, watts_per_busy_core: float = 6.0,
                 battery_capacity_wh: float | None = None) -> None:
        self.watts_per_busy_core = float(watts_per_busy_core)
        self._lock = threading.Lock()
        self.accounts: dict[str, Account] = {}
        self.readings = 0

        self._rapl = _rapl_domains()
        now_wh, full_wh = _battery_energy_wh()
        self._battery_available = now_wh is not None
        self.battery_capacity_wh = battery_capacity_wh or full_wh
        self.source = (Source.RAPL if self._rapl
                       else Source.BATTERY if self._battery_available
                       else Source.MODEL)
        # RAPL counters wrap; remember the last value so a wrap is corrected
        # rather than reported as a device that generated energy.
        self._last_rapl: dict[str, int] = {}
        self._rapl_range = self._read_rapl_range()

    # -- raw counters -----------------------------------------------------

    def _read_rapl_range(self) -> dict[str, int]:
        ranges: dict[str, int] = {}
        for counter in self._rapl:
            limit = counter.parent / "max_energy_range_uj"
            try:
                ranges[str(counter)] = int(limit.read_text().strip())
            except Exception:
                ranges[str(counter)] = 0
        return ranges

    def _rapl_microjoules(self) -> int | None:
        if not self._rapl:
            return None
        total = 0
        for counter in self._rapl:
            try:
                value = int(counter.read_text().strip())
            except Exception:
                return None
            key = str(counter)
            previous = self._last_rapl.get(key)
            self._last_rapl[key] = value
            if previous is None:
                continue
            delta = value - previous
            if delta < 0:                      # counter wrapped
                delta += self._rapl_range.get(key, 0)
            total += max(delta, 0)
        return total

    @staticmethod
    def _cpu_seconds() -> float:
        usage = os.times()
        return usage.user + usage.system + usage.children_user + usage.children_system

    # -- measurement ------------------------------------------------------

    def sample(self, kind: str, wall_seconds: float, cpu_seconds: float,
               ops: int = 1) -> Reading:
        """Turn an elapsed operation into a reading, from the best source there is."""
        joules = 0.0
        source = self.source
        micro = self._rapl_microjoules()
        if micro is not None and micro > 0:
            joules, source = micro / 1e6, Source.RAPL
        else:
            # Busy-core seconds times an assumed per-core draw. Stated, not hidden.
            joules, source = cpu_seconds * self.watts_per_busy_core, Source.MODEL
        reading = Reading(joules=joules, cpu_seconds=cpu_seconds,
                          wall_seconds=wall_seconds, source=source, ops=ops)
        with self._lock:
            self.readings += 1
            self.accounts.setdefault(kind, Account(kind)).add(reading)
        return reading

    class _Span:
        __slots__ = ("meter", "kind", "ops", "_t0", "_c0", "reading")

        def __init__(self, meter: "EnergyMeter", kind: str, ops: int) -> None:
            self.meter, self.kind, self.ops = meter, kind, ops
            self.reading: Reading | None = None

        def __enter__(self) -> "EnergyMeter._Span":
            self._t0 = time.perf_counter()
            self._c0 = EnergyMeter._cpu_seconds()
            self.meter._rapl_microjoules()          # prime the delta
            return self

        def __exit__(self, *_exc: Any) -> None:
            self.reading = self.meter.sample(
                self.kind,
                wall_seconds=time.perf_counter() - self._t0,
                cpu_seconds=EnergyMeter._cpu_seconds() - self._c0,
                ops=self.ops)

    def measure(self, kind: str, ops: int = 1) -> "_Span":
        """`with meter.measure("query"): ...` — costs two clock reads."""
        return EnergyMeter._Span(self, kind, ops)

    # -- what it all means ------------------------------------------------

    def per_battery_percent(self, kind: str) -> float | None:
        """How many of this operation fit in one percent of the pack.

        The sentence a person actually remembers. It needs a capacity, which
        this reads from the battery when there is one and otherwise takes from
        configuration — and returns None rather than inventing a pack that
        isn't there.
        """
        account = self.accounts.get(kind)
        if not account or not account.ops or self.battery_capacity_wh is None:
            return None
        joules_per_op = account.joules / account.ops
        if joules_per_op <= 0:
            return None
        one_percent_joules = self.battery_capacity_wh * 3600.0 * 0.01
        return one_percent_joules / joules_per_op

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            accounts = {k: a.as_dict() for k, a in self.accounts.items()}
        for kind in list(accounts):
            fits = self.per_battery_percent(kind)
            accounts[kind]["per_battery_percent"] = round(fits) if fits else None
        return {
            "source": self.source.value,
            "measured": self.source.measured,
            "how": {
                Source.RAPL.value: "CPU package energy counter, in microjoules",
                Source.BATTERY.value: "pack discharge over the window, whole device",
                Source.MODEL.value: (f"CPU-seconds x {self.watts_per_busy_core} W per busy "
                                     "core — a model, not a measurement"),
                Source.NONE.value: "no source available",
            }[self.source.value],
            "rapl_domains": len(self._rapl),
            "battery_capacity_wh": self.battery_capacity_wh,
            "watts_per_busy_core": self.watts_per_busy_core,
            "readings": self.readings,
            "by_kind": accounts,
        }
