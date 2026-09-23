"""Thermal and power governor.

When the package is hot or the battery is low, the node sheds inference cost
by swapping the embedder to a smaller variant instead of thermally throttling
into unresponsiveness. Quality degrades on a curve we chose, not one the
silicon chose for us.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass

from ..core.bus import EventBus
from ..core.metrics import METRICS

LADDER = ["fp32", "fp16", "int8-dynamic", "int8-static"]


@dataclass(slots=True)
class PowerState:
    temp_c: float = 52.0
    battery_pct: float = 88.0
    on_mains: bool = True
    cpu_load: float = 0.3


class ThermalGovernor:
    def __init__(self, bus: EventBus, embedder, ceiling_c: float, battery_floor_pct: float) -> None:
        self.bus = bus
        self.embedder = embedder
        self.ceiling_c = ceiling_c
        self.battery_floor_pct = battery_floor_pct
        self.state = PowerState()
        self.swaps = 0
        self.last_swap = 0.0
        self.forced: str | None = None

    def sample(self) -> PowerState:
        """Read sensors; synthesised where the platform exposes none."""
        try:
            import psutil  # type: ignore

            temps = psutil.sensors_temperatures() or {}
            readings = [t.current for group in temps.values() for t in group]
            if readings:
                self.state.temp_c = max(readings)
            battery = psutil.sensors_battery()
            if battery:
                self.state.battery_pct = battery.percent
                self.state.on_mains = battery.power_plugged
            self.state.cpu_load = psutil.cpu_percent(interval=None) / 100.0
        except Exception:
            drift = random.uniform(-1.4, 1.6)
            self.state.temp_c = max(38.0, min(96.0, self.state.temp_c + drift))
            if not self.state.on_mains:
                self.state.battery_pct = max(0.0, self.state.battery_pct - 0.05)
        METRICS.gauge("governor.temp_c", round(self.state.temp_c, 1))
        METRICS.gauge("governor.battery_pct", round(self.state.battery_pct, 1))
        return self.state

    def desired_variant(self) -> str:
        if self.forced:
            return self.forced
        if self.state.temp_c > self.ceiling_c + 8:
            return "int8-static"
        if self.state.temp_c > self.ceiling_c or self.state.battery_pct < self.battery_floor_pct:
            return "int8-dynamic"
        if self.state.temp_c > self.ceiling_c - 12:
            return "fp16"
        return "fp32" if self.state.on_mains else "fp16"

    def enforce(self) -> str | None:
        current = self.embedder.entry.variant
        target = self.desired_variant()
        if target == current or time.time() - self.last_swap < 5.0:  # hysteresis
            return None
        self.embedder.swap_variant(target)
        self.swaps += 1
        self.last_swap = time.time()
        METRICS.incr("governor.swaps")
        self.bus.publish(
            "telemetry", "variant_swap", level="warn",
            from_variant=current, to_variant=target, temp_c=round(self.state.temp_c, 1),
            message=f"governor swapped embedder <b>{current}</b> → <b>{target}</b> at {self.state.temp_c:.0f}°C",
        )
        return target

    def snapshot(self) -> dict[str, object]:
        return {
            "temp_c": round(self.state.temp_c, 1),
            "battery_pct": round(self.state.battery_pct, 1),
            "on_mains": self.state.on_mains,
            "variant": self.embedder.entry.variant,
            "swaps": self.swaps,
        }
