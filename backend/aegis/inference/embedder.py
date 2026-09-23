"""Dense embedder: registry-backed graph + micro-batching + metrics."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.metrics import METRICS
from .batcher import MicroBatcher
from .onnx_runtime import OnnxSession
from .registry import ModelEntry, ModelRegistry


class Embedder:
    role = "embedder"

    def __init__(self, registry: ModelRegistry, name: str, dim: int, window_ms: float, max_batch: int,
                 variant: str = "int8-dynamic", model_dir: str = "models") -> None:
        self.registry = registry
        self.dim = dim
        self.name = name
        path = Path(model_dir) / f"{name}.{variant}.onnx"
        self.entry = registry.register(self.role, ModelEntry(
            name=name, version="1", variant=variant,
            path=str(path) if path.exists() else None, dim=dim,
        ))
        self.session = OnnxSession(self.role, path if path.exists() else None, dim, Path(model_dir) / ".cache")
        self.entry.loaded = True
        self.batcher = MicroBatcher(self._encode, window_ms, max_batch)

    def _encode(self, texts: list[str]) -> np.ndarray:
        with METRICS.timer("inference.embed_ms"):
            return self.session.encode(texts)

    async def embed(self, text: str) -> np.ndarray:
        return await self.batcher.submit(text)

    def embed_sync(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def swap_variant(self, variant: str, model_dir: str = "models") -> None:
        """Hot-swap precision (thermal governor) without dropping the role."""
        path = Path(model_dir) / f"{self.name}.{variant}.onnx"
        entry = self.registry.register(self.role, ModelEntry(
            name=self.name, version="1", variant=variant,
            path=str(path) if path.exists() else None, dim=self.dim,
        ))
        self.session = OnnxSession(self.role, path if path.exists() else None, self.dim,
                                   Path(model_dir) / ".cache")
        self.entry = entry
        self.batcher = MicroBatcher(self._encode, self.batcher.window_s * 1000, self.batcher.max_batch)

    @property
    def version(self) -> str:
        return f"{self.name}@{self.entry.version}"

    def snapshot(self) -> dict[str, object]:
        hist = METRICS.histograms.get("inference.embed_ms")
        return {
            "model": self.name,
            "variant": self.entry.variant,
            "dim": self.dim,
            "session": self.session.snapshot(),
            "batching": self.batcher.snapshot(),
            "latency_ms": hist.snapshot() if hist else {},
        }
