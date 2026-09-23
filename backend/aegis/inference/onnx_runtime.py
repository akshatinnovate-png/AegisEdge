"""ONNX Runtime sessions and the execution-provider ladder.

Real weights, real tokenizer, real graphs. There is no synthetic encoder
behind this: if the model cannot be provisioned the node refuses to start,
because an edge device that silently degrades to a toy encoder produces
confident nonsense and no one is watching to catch it.

The provider ladder is probed at boot and the best available accelerator is
taken. The optimized graph is serialized on first load so a cold start is a
load rather than a re-optimization — which on a device that reboots whenever
the vehicle does is the difference between ready in 200 ms and ready in ten
seconds.
"""
from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any

import numpy as np

from .models import ModelsUnavailable, provision

EP_LADDER = [
    ("TensorrtExecutionProvider", "TENSORRT"),
    ("CUDAExecutionProvider", "CUDA"),
    ("ROCMExecutionProvider", "ROCM"),
    ("MIGraphXExecutionProvider", "MIGRAPHX"),
    ("OpenVINOExecutionProvider", "OPENVINO"),
    ("CoreMLExecutionProvider", "COREML"),
    ("NnapiExecutionProvider", "NNAPI"),
    ("QNNExecutionProvider", "QNN"),
    ("XnnpackExecutionProvider", "XNNPACK"),
    ("DnnlExecutionProvider", "DNNL"),
    ("AzureExecutionProvider", None),          # remote: never selected on an edge node
    ("CPUExecutionProvider", "CPU"),
]

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Word-level split, used by the lexical helpers — not by the model."""
    return _TOKEN.findall(text.lower())


def probe_providers() -> tuple[str, list[str]]:
    """Walk the ladder and report the best provider this device actually has."""
    try:
        import onnxruntime as ort
    except ImportError as exc:                    # pragma: no cover - dependency guard
        raise ModelsUnavailable("onnxruntime is required") from exc
    available = list(ort.get_available_providers())
    for provider, label in EP_LADDER:
        if label and provider in available:
            return label, available
    return "CPU", available


class Tokenizer:
    """The model's own BPE tokenizer, shared across sessions."""

    _lock = threading.Lock()
    _cache: dict[str, "Tokenizer"] = {}

    def __init__(self, path: Path, max_tokens: int = 128) -> None:
        from tokenizers import Tokenizer as HFTokenizer

        self.path = Path(path)
        self.max_tokens = max_tokens
        self._tokenizer = HFTokenizer.from_file(str(path))
        self.vocab = self._tokenizer.get_vocab_size()

    @classmethod
    def load(cls, path: Path, max_tokens: int = 128) -> "Tokenizer":
        key = f"{path}:{max_tokens}"
        with cls._lock:
            if key not in cls._cache:
                cls._cache[key] = cls(path, max_tokens)
            return cls._cache[key]

    def encode(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Batch to padded ids + mask. Padding is masked, never averaged in."""
        rows = [self._tokenizer.encode(text).ids[: self.max_tokens] or [0] for text in texts]
        width = max(len(row) for row in rows)
        ids = np.zeros((len(rows), width), dtype=np.int64)
        mask = np.zeros((len(rows), width), dtype=np.float32)
        for index, row in enumerate(rows):
            ids[index, : len(row)] = row
            mask[index, : len(row)] = 1.0
        return ids, mask

    def token_ids(self, text: str) -> list[int]:
        return self._tokenizer.encode(text).ids[: self.max_tokens]

    def token_strings(self, text: str) -> list[str]:
        return self._tokenizer.encode(text).tokens[: self.max_tokens]


class OnnxSession:
    """One graph, one provider, real inference."""

    def __init__(self, role: str, model_path: Path, tokenizer_path: Path, dim: int,
                 cache_dir: Path, max_tokens: int = 128) -> None:
        import onnxruntime as ort

        self.role = role
        self.dim = dim
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise ModelsUnavailable(f"no graph for role '{role}' at {model_path}")

        self.provider, self.available = probe_providers()
        self.tokenizer = Tokenizer.load(Path(tokenizer_path), max_tokens)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = cache_dir / f"{role}.optimized.onnx"

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 0
        options.enable_mem_pattern = True
        if not self._cache.exists():
            options.optimized_model_filepath = str(self._cache)   # AOT: pay once, not per boot
        providers = [p for p, label in EP_LADDER if label and p in self.available]
        source = self._cache if self._cache.exists() else self.model_path
        self.session = ort.InferenceSession(str(source), options, providers=providers)
        self.active_provider = self.session.get_providers()[0]
        self.runs = 0
        self.tokens_seen = 0

    # -- inference --------------------------------------------------------

    def encode(self, texts: list[str]) -> np.ndarray:
        """Sentence embeddings: (batch, dim), L2-normalised by the graph itself."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        ids, mask = self.tokenizer.encode(texts)
        self.runs += 1
        self.tokens_seen += int(mask.sum())
        return self.session.run(None, {"input_ids": ids, "attention_mask": mask})[0]

    def token_embeddings(self, text: str) -> tuple[np.ndarray, list[str]]:
        """Per-token vectors for late interaction, with their surface forms."""
        ids = self.tokenizer.token_ids(text)
        if not ids:
            return np.zeros((0, self.dim), dtype=np.float32), []
        array = np.asarray([ids], dtype=np.int64)
        self.runs += 1
        self.tokens_seen += len(ids)
        vectors = self.session.run(None, {"input_ids": array})[0][0]
        return vectors, self.tokenizer.token_strings(text)

    def snapshot(self) -> dict[str, Any]:
        return {
            "role": self.role, "provider": self.provider,
            "active_provider": self.active_provider,
            "graph": str(self.model_path), "optimized_cache": self._cache.exists(),
            "dim": self.dim, "vocab": self.tokenizer.vocab,
            "runs": self.runs, "tokens": self.tokens_seen,
        }


def open_sessions(model_dir: Path, max_tokens: int = 128) -> tuple[OnnxSession, OnnxSession, Any]:
    """Provision the real weights and open both graphs."""
    bundle = provision(Path(model_dir))
    cache = Path(model_dir) / ".cache"
    embedder = OnnxSession("embedder", bundle.embedder_path, bundle.tokenizer_path,
                           bundle.dim, cache, max_tokens)
    reranker = OnnxSession("reranker", bundle.reranker_path, bundle.tokenizer_path,
                           bundle.dim, cache, max_tokens)
    return embedder, reranker, bundle
