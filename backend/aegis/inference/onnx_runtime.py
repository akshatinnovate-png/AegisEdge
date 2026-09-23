"""ONNX Runtime session management and the execution-provider ladder.

The node probes for accelerators at boot and takes the best one available,
serializing the optimized graph so a cold start is a load rather than a
re-optimization. Where onnxruntime or the graph itself is absent, a
deterministic hashed-n-gram encoder stands in so every downstream subsystem
still runs end to end — and the fallback is reported, never hidden.
"""
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

EP_LADDER = [
    ("TensorrtExecutionProvider", "TENSORRT"),
    ("CUDAExecutionProvider", "CUDA"),
    ("ROCMExecutionProvider", "ROCM"),
    ("OpenVINOExecutionProvider", "OPENVINO"),
    ("CoreMLExecutionProvider", "COREML"),
    ("NnapiExecutionProvider", "NNAPI"),
    ("XnnpackExecutionProvider", "XNNPACK"),
    ("CPUExecutionProvider", "CPU"),
]

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def probe_providers() -> tuple[str, list[str]]:
    """Return (chosen label, available providers) walking the ladder in order."""
    try:
        import onnxruntime as ort  # type: ignore

        available = list(ort.get_available_providers())
    except Exception:
        return "XNNPACK (CPU, fallback encoder)", []
    for provider, label in EP_LADDER:
        if provider in available:
            return label, available
    return "CPU", available


class OnnxSession:
    """Wraps one graph. Falls back to the hashed encoder when unavailable."""

    def __init__(self, role: str, model_path: Path | None, dim: int, cache_dir: Path) -> None:
        self.role = role
        self.dim = dim
        self.model_path = model_path
        self.provider, self.available = probe_providers()
        self.session: Any | None = None
        self.fallback = True
        self.runs = 0
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = cache_dir / f"{role}.optimized.onnx"
        if model_path is not None and Path(model_path).exists():
            self._open(Path(model_path))

    def _open(self, path: Path) -> None:
        try:
            import onnxruntime as ort  # type: ignore

            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.intra_op_num_threads = 0
            if not self._cache.exists():
                options.optimized_model_filepath = str(self._cache)   # AOT serialize
            providers = [p for p, _ in EP_LADDER if p in self.available]
            self.session = ort.InferenceSession(str(path), options, providers=providers)
            self.fallback = False
        except Exception:
            self.session = None
            self.fallback = True

    # -- encoding ---------------------------------------------------------

    def encode(self, texts: list[str]) -> np.ndarray:
        self.runs += 1
        if self.session is not None:
            try:
                return self._encode_onnx(texts)
            except Exception:
                self.fallback = True
        return self._encode_hashed(texts)

    def _encode_onnx(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - needs a real graph
        inputs = self.session.get_inputs()
        ids = np.array([[abs(hash(t)) % 30000 for t in tokenize(x)][:128] or [0] for x in texts], dtype=np.int64)
        feed = {inputs[0].name: ids}
        if len(inputs) > 1:
            feed[inputs[1].name] = np.ones_like(ids)
        out = self.session.run(None, feed)[0]
        vectors = out.mean(axis=1) if out.ndim == 3 else out
        return self._normalize(np.asarray(vectors, dtype=np.float32))

    def _encode_hashed(self, texts: list[str]) -> np.ndarray:
        """Deterministic hashed word+bigram encoder with sublinear TF.

        Not a transformer — but stable, dependency-free and semantically
        ordered enough that retrieval, fusion, renewal and sync are all
        exercised for real.
        """
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = tokenize(text)
            grams = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
            for gram in grams:
                digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(digest[:4], "big") % self.dim
                sign = 1.0 if digest[4] & 1 else -1.0
                out[row, idx] += sign * (1.0 + math.log1p(len(gram)))
        return self._normalize(out)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vectors / norms).astype(np.float32)

    def snapshot(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "provider": self.provider,
            "graph": str(self.model_path) if self.model_path else None,
            "fallback_encoder": self.fallback,
            "runs": self.runs,
        }
