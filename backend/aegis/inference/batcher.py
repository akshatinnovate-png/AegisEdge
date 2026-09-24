"""Dynamic micro-batching.

Concurrent embed calls inside an 8 ms window are coalesced into one session
run. Burst ingest is where an edge node actually spends its cycles, and one
padded batch beats eight sequential runs by roughly 4x.

The batch runs on the event loop, and that is deliberate — it was tried the
other way. The obvious suspicion is that holding the loop for the duration of
a session run serialises the whole node, so `_flush_now` was moved onto a
`ThreadPoolExecutor`. Measured on unique queries at 256 concurrent:

    synchronous, on the loop     2,069 qps    p50  62.97 ms
    handed to a worker pool      1,332 qps    p50 108.84 ms

Worse, and not marginally. ONNX Runtime already releases the GIL for the
duration of `run()` and parallelises internally with its own intra-op pool, so
the executor added a hop per batch and bought nothing back. Pool width made no
difference either (1,480 / 1,503 / 1,531 qps at one, two and four workers),
which is what you would expect if the threads were never the constraint.

What *is* the constraint is the pipeline's own Python: fusion, scoring,
diversity and conformal calibration all run in the interpreter, and one core
of interpreter is the ceiling. The lever that moved is upstream of here — the
semantic cache, which had been disabled by an inferred filter and now answers
a repeated question before the encoder is ever reached.
"""
from __future__ import annotations

import asyncio
import time
from typing import Callable

import numpy as np

from ..core.metrics import METRICS


class MicroBatcher:
    def __init__(self, fn: Callable[[list[str]], np.ndarray], window_ms: float = 8.0, max_batch: int = 32) -> None:
        self._fn = fn
        self.window_s = window_ms / 1000.0
        self.max_batch = max_batch
        self._pending: list[tuple[str, asyncio.Future]] = []
        self._flush_task: asyncio.Task | None = None
        self.batches = 0
        self.items = 0
        self.largest = 0

    async def submit(self, text: str) -> np.ndarray:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending.append((text, future))
        if len(self._pending) >= self.max_batch:
            self._flush_now()
        elif self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._flush_after_window())
        return await future

    async def _flush_after_window(self) -> None:
        await asyncio.sleep(self.window_s)
        self._flush_now()

    def _flush_now(self) -> None:
        if not self._pending:
            return
        batch, self._pending = self._pending[: self.max_batch], self._pending[self.max_batch:]
        texts = [t for t, _ in batch]
        t0 = time.perf_counter()
        try:
            vectors = self._fn(texts)
        except Exception as exc:  # propagate to every waiter, never hang them
            for _, fut in batch:
                if not fut.done():
                    fut.set_exception(exc)
            return
        elapsed = (time.perf_counter() - t0) * 1000.0
        self.batches += 1
        self.items += len(batch)
        self.largest = max(self.largest, len(batch))
        METRICS.observe("inference.batch_ms", elapsed)
        METRICS.observe("inference.batch_size", len(batch))
        for (_, fut), vector in zip(batch, vectors):
            if not fut.done():
                fut.set_result(vector)
        if self._pending:
            self._flush_now()

    def snapshot(self) -> dict[str, float]:
        return {
            "batches": self.batches,
            "items": self.items,
            "largest_batch": self.largest,
            "avg_batch": round(self.items / self.batches, 2) if self.batches else 0.0,
        }
