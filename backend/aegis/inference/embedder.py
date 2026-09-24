"""Dense embedder: real ONNX graph, micro-batched, with metrics."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..core.metrics import METRICS
from .adaptation import AdaptationGate
from .batcher import MicroBatcher
from .geometry import CorpusGeometry
from .lexicon import TokenLexicon
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

        # Two on-device adaptations, both off until they have earned their way
        # on. The lexicon learns which tokens this device's corpus actually
        # discriminates by and reweights pooling accordingly; the geometry
        # measures the corpus covariance and corrects a pretrained space that
        # is anisotropic as shipped. Neither is a guess: `scripts/geometry_eval.py`
        # scores them against paraphrase retrieval with known ground truth, and
        # the node refuses to arm either one on the strength of theory alone.
        self.lexicon = TokenLexicon(session.tokenizer.vocab)
        self.geometry = CorpusGeometry(self.dim)
        self.gate = AdaptationGate()
        self.adapted = 0

    def _encode(self, texts: list[str]) -> np.ndarray:
        with METRICS.timer("inference.embed_ms"):
            vectors = self.session.encode(texts)
        if self.geometry.enabled:
            vectors = self.geometry.transform(vectors)
            self.adapted += len(texts)
        return vectors

    # -- on-device adaptation ---------------------------------------------

    def observe(self, texts: list[str]) -> None:
        """Feed the corpus statistics. Cheap, and never on the critical path.

        Called from the maintenance lane with text the node has already
        ingested, so a write never pays for it.
        """
        if not texts:
            return
        self.lexicon.observe(self.session.tokenizer.token_ids(t) for t in texts)

    def observe_vectors(self, vectors: np.ndarray) -> None:
        self.geometry.observe(vectors)

    def enable_weighted_pooling(self, on: bool = True) -> bool:
        """Arm SIF pooling. Changes the embedding space, so it is versioned."""
        if on and not self.lexicon.informed:
            return False
        self.session.weighting = self.lexicon if on else None
        return self.session.weighting is not None

    def self_evaluate(self, texts: list[str], queries: int = 200,
                      arm: bool = False) -> dict[str, Any]:
        """Score the adaptations against the shipped space and arm the winners.

        Runs on the maintenance lane over text the node has already stored.
        The whole pass is a few hundred embeddings — milliseconds at this
        model's throughput — so a device can afford to re-check its own
        representation as its corpus drifts, instead of inheriting a decision
        somebody made once on somebody else's data.

        **Measuring never arms anything.** A rank-limited whitening transform
        changes the output dimension — 256 to 195 on the corpus this was
        developed against — so switching it on mid-flight would leave every
        stored vector in a space of a different shape to the queries hunting
        it. That is not a degradation, it is nonsense, and no retrieval metric
        would report it. So this returns a *recommendation*, and `arm=True` is
        for the renewal path that has a migration to run behind it.
        """
        if len(texts) < self.gate.min_documents:
            return {"ran": False, "reason": f"corpus too small: {len(texts)}"}

        baseline_weighting = self.session.weighting
        self.session.weighting = None
        was_enabled = self.geometry.enabled
        self.geometry.enable(False)
        try:
            def encode(batch: list[str]) -> np.ndarray:
                return np.vstack([self.session.encode(batch[i:i + self._max_batch])
                                  for i in range(0, len(batch), self._max_batch)])

            sample = texts[: max(self.gate.min_documents * 4, 2000)]
            probe_vectors = encode(sample[: min(len(sample), 2000)])
            fitted = CorpusGeometry(self.dim)
            fitted.observe(probe_vectors)
            version = fitted.fit()

            candidates: list[tuple[str, Any]] = []
            if version is not None:
                candidates.append((f"corpus whitening (rank {fitted.rank})", fitted.transform))
            report = self.gate.evaluate(sample, encode, candidates, queries=queries)
        finally:
            self.session.weighting = baseline_weighting
            self.geometry.enable(was_enabled)

        decision = report.get("decision") or {}
        if decision.get("adopted") and version is not None:
            report["recommendation"] = {
                "geometry": version.as_dict(),
                "current_dim": self.dim,
                "proposed_dim": fitted.out_dim,
                "dimension_change": fitted.out_dim != self.dim,
                "migration_required": True,
                "note": ("re-embed every stored point into the new space before "
                         "arming; a vector from one space and a query from "
                         "another are not comparable"),
            }
            if arm:
                self.geometry = fitted
                self.geometry.enable(True)
                report["armed"] = {"space_version": self.space_version,
                                   "out_dim": fitted.out_dim}
        return report

    def arm_geometry(self, on: bool = True) -> dict[str, Any]:
        """Switch the fitted transform on or off. Callers own the migration."""
        enabled = self.geometry.enable(on)
        return {"enabled": enabled, "space_version": self.space_version,
                "out_dim": self.geometry.out_dim if enabled else self.dim}

    @property
    def space_version(self) -> str:
        """Identity of the space this embedder currently produces.

        Pooling weights and a whitening transform both change what a vector
        *means*. A vector produced under one and compared against another is
        not slightly wrong, it is meaningless — so the identity travels with
        the vector and the renewal migrator refuses to mix them.
        """
        parts = [self.name, f"v{self.entry.version}"]
        if self.session.weighting is not None:
            parts.append(f"sif{self.lexicon.version}")
        if self.geometry.enabled and self.geometry.version is not None:
            parts.append(f"geo{self.geometry.version.id}")
        return "+".join(parts)

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
            "space_version": self.space_version,
            "lexicon": self.lexicon.snapshot(),
            "geometry": self.geometry.snapshot(),
            "adaptation_gate": self.gate.snapshot(),
            "adapted_vectors": self.adapted,
        }
