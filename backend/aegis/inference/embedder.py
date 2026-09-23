"""Dense embedder: real ONNX graph, micro-batched, with metrics."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..core.metrics import METRICS
from .batcher import MicroBatcher
from .onnx_runtime import OnnxSession
from .registry import ModelEntry, ModelRegistry


class Embedder:
    role = "embedder"

    def __init__(self, registry: ModelRegistry, session: OnnxSession, bundle,
                 window_ms: float, max_batch: int) -> None:
        self.registry = registry
        self.session = session
        self.bundle = bundle
        self.dim = session.dim
        self.name = bundle.source
        self.entry = registry.register(self.role, ModelEntry(
            name=self.name, version="1", variant="fp32",
            path=str(bundle.embedder_path), sha256=bundle.sha256, dim=self.dim,
            meta={"vocab": bundle.vocab, "provenance": bundle.source},
        ))
        self.entry.loaded = True
        self.batcher = MicroBatcher(self._encode, window_ms, max_batch)
        self._window_ms = window_ms
        self._max_batch = max_batch

    def _encode(self, texts: list[str]) -> np.ndarray:
        with METRICS.timer("inference.embed_ms"):
            return self.session.encode(texts)

    async def embed(self, text: str) -> np.ndarray:
        return await self.batcher.submit(text)

    def embed_sync(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def set_precision(self, variant: str) -> None:
        """Record the precision the governor selected.

        The graph is a gather plus a pooling reduction: there is no separate
        quantized artefact to swap, and claiming one would be theatre. What
        the governor actually controls here is the batching window and the
        token budget, which is where the cost is.
        """
        self.entry = self.registry.register(self.role, ModelEntry(
            name=self.name, version="1", variant=variant,
            path=str(self.bundle.embedder_path), sha256=self.bundle.sha256, dim=self.dim,
        ))
        self.entry.loaded = True

    def set_token_budget(self, max_tokens: int) -> None:
        """Shed inference cost by truncating inputs, not by faking a swap."""
        self.session.tokenizer.max_tokens = max(8, int(max_tokens))

    @property
    def version(self) -> str:
        return f"{self.name}@{self.entry.version}"

    def snapshot(self) -> dict[str, Any]:
        hist = METRICS.histograms.get("inference.embed_ms")
        return {
            "model": self.name, "variant": self.entry.variant, "dim": self.dim,
            "max_tokens": self.session.tokenizer.max_tokens,
            "session": self.session.snapshot(), "batching": self.batcher.snapshot(),
            "latency_ms": hist.snapshot() if hist else {},
        }
