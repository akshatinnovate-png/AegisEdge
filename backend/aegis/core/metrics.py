"""Counters, gauges and streaming-quantile histograms + Prometheus exposition."""
from __future__ import annotations

import bisect
import threading
from collections import defaultdict
from typing import Any


class Histogram:
    __slots__ = ("name", "_window", "_samples", "count", "total")

    def __init__(self, name: str, window: int = 1024) -> None:
        self.name = name
        self._window = window
        self._samples: list[float] = []
        self.count = 0
        self.total = 0.0

    def observe(self, value: float) -> None:
        self.count += 1
        self.total += value
        bisect.insort(self._samples, value)
        if len(self._samples) > self._window:
            self._samples.pop(len(self._samples) // 2)

    def quantile(self, q: float) -> float:
        if not self._samples:
            return 0.0
        idx = min(len(self._samples) - 1, int(q * len(self._samples)))
        return self._samples[idx]

    def snapshot(self) -> dict[str, float]:
        return {
            "count": self.count,
            "mean": round(self.total / self.count, 4) if self.count else 0.0,
            "p50": round(self.quantile(0.50), 4),
            "p95": round(self.quantile(0.95), 4),
            "p99": round(self.quantile(0.99), 4),
        }


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counters: dict[str, float] = defaultdict(float)
        self.gauges: dict[str, float] = {}
        self.histograms: dict[str, Histogram] = {}

    def incr(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self.counters[name] += value

    def gauge(self, name: str, value: float) -> None:
        self.gauges[name] = value

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            hist = self.histograms.get(name)
            if hist is None:
                hist = self.histograms[name] = Histogram(name)
        hist.observe(value)

    def timer(self, name: str) -> "Timer":
        return Timer(self, name)

    def snapshot(self) -> dict[str, Any]:
        return {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histograms": {k: v.snapshot() for k, v in self.histograms.items()},
        }

    def prometheus(self) -> str:
        lines: list[str] = []
        for name, value in sorted(self.counters.items()):
            metric = name.replace(".", "_").replace("-", "_")
            lines.append(f"# TYPE aegis_{metric} counter")
            lines.append(f"aegis_{metric} {value}")
        for name, value in sorted(self.gauges.items()):
            metric = name.replace(".", "_").replace("-", "_")
            lines.append(f"# TYPE aegis_{metric} gauge")
            lines.append(f"aegis_{metric} {value}")
        for name, hist in sorted(self.histograms.items()):
            metric = name.replace(".", "_").replace("-", "_")
            lines.append(f"# TYPE aegis_{metric} summary")
            for q in (0.5, 0.95, 0.99):
                lines.append(f'aegis_{metric}{{quantile="{q}"}} {hist.quantile(q)}')
            lines.append(f"aegis_{metric}_count {hist.count}")
            lines.append(f"aegis_{metric}_sum {hist.total}")
        return "\n".join(lines) + "\n"


class Timer:
    __slots__ = ("_metrics", "_name", "_t0", "elapsed_ms")

    def __init__(self, metrics: Metrics, name: str) -> None:
        self._metrics = metrics
        self._name = name
        self.elapsed_ms = 0.0

    def __enter__(self) -> "Timer":
        import time
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        import time
        self.elapsed_ms = (time.perf_counter() - self._t0) * 1000.0
        self._metrics.observe(self._name, self.elapsed_ms)


METRICS = Metrics()
