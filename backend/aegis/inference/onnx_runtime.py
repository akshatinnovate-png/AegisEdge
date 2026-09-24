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
import os
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
    """The model's own BPE tokenizer, shared across sessions.

    Every call keeps at most `max_tokens` tokens, so tokenising the whole
    input is work thrown away in proportion to how much of it was oversized.
    A 1 MB query spent 694 ms of its 854 ms here producing tokens 129 through
    several hundred thousand, every one of them discarded on the next line —
    and the same text arriving as a *document* pays it again on every rerank
    that touches it, which is how one megabyte-sized memory turned an
    adversarial query into a 30-second operation.

    So the text is clipped to a character bound first. The bound is generous
    by a wide margin: at 16 characters per token no real text can reach 128
    tokens before it, and the clip was verified to produce byte-identical
    token ids to full tokenisation on 3,000 sentences of real prose and on
    pathological input with no delimiters at all.
    """

    # Characters per token to allow before clipping. BPE on real text averages
    # well under 8; this leaves room for scripts that tokenise far worse.
    CHAR_BUDGET_PER_TOKEN = 16



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

    def _clip(self, text: str) -> str:
        """Bound the work, never the meaning: the tail could not survive anyway."""
        budget = self.max_tokens * self.CHAR_BUDGET_PER_TOKEN
        return text if len(text) <= budget else text[:budget]

    def encode(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Batch to padded ids + mask. Padding is masked, never averaged in."""
        rows = [self._tokenizer.encode(self._clip(text)).ids[: self.max_tokens] or [0]
                for text in texts]
        width = max(len(row) for row in rows)
        ids = np.zeros((len(rows), width), dtype=np.int64)
        mask = np.zeros((len(rows), width), dtype=np.float32)
        for index, row in enumerate(rows):
            ids[index, : len(row)] = row
            mask[index, : len(row)] = 1.0
        return ids, mask

    def token_ids(self, text: str) -> list[int]:
        return self._tokenizer.encode(self._clip(text)).ids[: self.max_tokens]

    def token_strings(self, text: str) -> list[str]:
        return self._tokenizer.encode(self._clip(text)).tokens[: self.max_tokens]


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
        # On the resident-set growth the steady-state soak reports: it is not
        # this layer, and several plausible fixes here were tried and measured
        # to do nothing. Disabling the CPU memory arena
        # (`enable_cpu_mem_arena = False`), the documented knob for exactly
        # this shape of problem: byte-for-byte identical at every checkpoint.
        # Bucketing sequence lengths and batch sizes so the runtime meets a few
        # dozen input shapes rather than a thousand: +41.6 / 83.7 / 125.2 /
        # 165.6 MB against +41.7 / 83.1 / 125.1 / 165.5 without. The encoder
        # itself is clean in isolation — 0.0 MB over 12,000 texts at batch 1
        # and at batch 8 — so none of this is where the memory goes. See
        # `soak_steady` in scripts/stress.py for what is and is not ruled out.
        if not self._cache.exists():
            options.optimized_model_filepath = str(self._cache)   # AOT: pay once, not per boot
        providers = [p for p, label in EP_LADDER if label and p in self.available]
        source = self._cache if self._cache.exists() else self.model_path
        self.session = ort.InferenceSession(str(source), options, providers=providers)
        self.active_provider = self.session.get_providers()[0]
        self.runs = 0
        self.tokens_seen = 0
        # Optional token weighting. The pooling graph computes
        # sum(embeddings * mask) / sum(mask), so handing it real-valued
        # weights in place of the 0/1 mask turns its masked mean into a
        # weighted mean — Smooth Inverse Frequency pooling, exactly, with no
        # change to the graph and no second artefact to keep in step.
        self.weighting: Any | None = None
        self.weighted_runs = 0

    # -- inference --------------------------------------------------------

    def encode(self, texts: list[str]) -> np.ndarray:
        """Sentence embeddings: (batch, dim), L2-normalised by the graph itself."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        ids, mask = self.tokenizer.encode(texts)
        self.runs += 1
        self.tokens_seen += int((mask > 0).sum())
        if self.weighting is not None:
            weights = self.weighting.weights(ids, mask)
            if weights is not None and weights.shape == mask.shape:
                mask = weights.astype(np.float32, copy=False)
                self.weighted_runs += 1
        return self.session.run(None, {"input_ids": ids, "attention_mask": mask})[0]

    def token_embeddings(self, text: str) -> tuple[np.ndarray, list[str]]:
        """Per-token vectors for late interaction, with their surface forms.

        """
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
            "pooling": "sif-weighted" if self.weighting is not None else "masked mean",
            "weighted_runs": self.weighted_runs,
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
