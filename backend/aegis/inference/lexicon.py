"""On-device token statistics: what this device has actually seen.

Mean pooling treats every token as equally informative. It is not. In a
maintenance corpus the token ``the`` carries no retrieval signal and the token
``spalling`` carries almost all of it, and averaging them with equal weight
lets a hundred function words outvote the one word the query was about.

Smooth Inverse Frequency (Arora, Liang & Ma, ICLR 2017) fixes this with a
weight per token,

    w(t) = a / (a + p(t))

where ``p(t)`` is the unigram probability of the token. Rare tokens approach
weight 1, common tokens fall towards ``a / p(t)``. The paper shows this alone
beats mean pooling on sentence similarity by a wide margin, and it beats
supervised models trained for the task — which is a remarkable result for
something that costs one multiply per token.

What makes it *edge* rather than borrowed: ``p(t)`` is estimated here, from
this device's own ingest stream, not shipped from a web crawl. A device on a
packing line and a device in a clinic converge on different weights because
they see different words, and neither needed a network to learn it.

Cold start is exact, not approximate: with no observations every weight is
1.0, which is precisely the masked mean the graph already computed. The model
degrades to its previous self rather than to noise.

**Measured result on this encoder: SIF pooling makes retrieval worse.** On
8,000 sentences of real prose, MRR@10 fell from 0.7827 to 0.7457 at a=1e-3 and
to 0.6376 at a=1e-4 — monotonically worse the more weight the reweighting
carried. The reason is specific and worth stating: this model's token table is
a distillation trained *for* mean pooling, so the frequency correction SIF
applies has already been folded into the weights, and applying it again
double-counts it. SIF remains the right tool for raw word vectors, which this
is not.

So the weighting ships off, and it is `AdaptationGate` — not this docstring
and not the citation — that decides whether any device ever turns it on. The
lexicon itself stays useful regardless: it is the node's own unigram
distribution, which the sparse retriever and the diagnostics both want.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# Arora et al. report the method is flat across a ∈ [1e-4, 1e-3]; we default to
# the midpoint and let `calibrate` move it to suit the observed distribution.
DEFAULT_A = 3e-4


class TokenLexicon:
    """Streaming unigram statistics over the model's own vocabulary.

    The vocabulary is 32k entries, so exact counts cost 256 KB — there is no
    reason to reach for a sketch and lose exactness to save nothing.
    """

    __slots__ = ("vocab", "counts", "documents", "tokens", "a", "min_observations",
                 "_lock", "version", "_dirty")

    def __init__(self, vocab: int, a: float = DEFAULT_A,
                 min_observations: int = 20_000) -> None:
        self.vocab = int(vocab)
        self.counts = np.zeros(self.vocab, dtype=np.int64)
        self.documents = 0
        self.tokens = 0
        self.a = float(a)
        # Below this many observed tokens the estimate of p(t) is noise, and
        # noisy weights are worse than no weights. Until then: uniform.
        self.min_observations = int(min_observations)
        self.version = 0
        self._dirty = False
        self._lock = threading.Lock()

    # -- learning ---------------------------------------------------------

    def observe(self, batches: Iterable[Iterable[int]]) -> int:
        """Fold a batch of tokenised documents into the counts."""
        seen = 0
        with self._lock:
            for ids in batches:
                array = np.asarray(list(ids), dtype=np.int64)
                if array.size == 0:
                    continue
                array = array[(array >= 0) & (array < self.vocab)]
                if array.size == 0:
                    continue
                np.add.at(self.counts, array, 1)
                self.documents += 1
                self.tokens += int(array.size)
                seen += int(array.size)
            if seen:
                self._dirty = True
        return seen

    @property
    def informed(self) -> bool:
        """Has this device seen enough of its own language to trust the estimate?"""
        return self.tokens >= self.min_observations

    def calibrate(self) -> float:
        """Set ``a`` to the median unigram probability of observed tokens.

        The reasoning is that ``a`` is the pivot of the weighting curve — a
        token with p(t) == a gets weight exactly 0.5 — so placing it at the
        median damps half the vocabulary and amplifies half, which carries the
        most information for a given corpus.

        The reasoning is also wrong, on this model, and the measurement says
        so: this heuristic picks a=2.3e-5 on real prose, which scored MRR@10
        0.5825 against 0.7827 for no weighting at all — the worst of every
        setting tried. It is kept because ``AdaptationGate`` sweeps ``a`` as
        one candidate among several and needs to be able to set it, and
        because a heuristic that lost to measurement is worth leaving visible
        next to the measurement that beat it.
        """
        with self._lock:
            observed = self.counts[self.counts > 0]
            if observed.size == 0 or self.tokens == 0:
                return self.a
            median = float(np.median(observed)) / float(self.tokens)
            # Clamp into the range the paper reports as stable.
            self.a = float(min(max(median, 1e-5), 1e-2))
            self.version += 1
            return self.a

    # -- use --------------------------------------------------------------

    def probabilities(self, ids: np.ndarray) -> np.ndarray:
        total = max(self.tokens, 1)
        flat = np.clip(np.asarray(ids, dtype=np.int64), 0, self.vocab - 1)
        return self.counts[flat].astype(np.float32) / float(total)

    def weights(self, ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """SIF weights shaped like the mask, with padding still exactly zero.

        The pooling graph divides by the sum of this array, so returning
        weights in place of a 0/1 mask turns its masked mean into a weighted
        mean with no change to the graph at all. Padding must stay at zero or
        it re-enters both the numerator and the denominator.
        """
        mask = np.asarray(mask, dtype=np.float32)
        if not self.informed:
            return mask
        weights = self.a / (self.a + self.probabilities(ids))
        weights *= mask
        # A document of nothing but stop words must still produce a vector;
        # if every weight collapsed, fall back to the plain mask for that row.
        totals = weights.sum(axis=1, keepdims=True)
        dead = (totals <= 1e-6).reshape(-1)
        if dead.any():
            weights[dead] = mask[dead]
        return weights.astype(np.float32, copy=False)

    # -- persistence ------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        with self._lock:
            # A path makes savez append ".npz"; a handle does not, and the
            # rename below has to find the file that was actually written.
            with open(temp, "wb") as handle:
                np.savez_compressed(handle, counts=self.counts,
                                    meta=np.frombuffer(json.dumps({
                                        "vocab": self.vocab, "documents": self.documents,
                                        "tokens": self.tokens, "a": self.a,
                                        "version": self.version,
                                    }).encode(), dtype=np.uint8))
            temp.replace(path)
            self._dirty = False

    @classmethod
    def load(cls, path: Path) -> "TokenLexicon | None":
        path = Path(path)
        if not path.exists():
            return None
        try:
            with np.load(path) as bundle:
                meta = json.loads(bytes(bundle["meta"]).decode())
                lexicon = cls(meta["vocab"], meta.get("a", DEFAULT_A))
                lexicon.counts = bundle["counts"].astype(np.int64)
                lexicon.documents = int(meta.get("documents", 0))
                lexicon.tokens = int(meta.get("tokens", 0))
                lexicon.version = int(meta.get("version", 0))
            return lexicon
        except Exception:
            # A corrupt lexicon is a cache miss, never a boot failure: the
            # model is exactly as correct without it, only less sharp.
            return None

    @property
    def dirty(self) -> bool:
        return self._dirty

    def top_tokens(self, n: int = 10) -> list[tuple[int, int]]:
        order = np.argsort(-self.counts)[:n]
        return [(int(i), int(self.counts[i])) for i in order if self.counts[i] > 0]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            distinct = int((self.counts > 0).sum())
            coverage = distinct / self.vocab if self.vocab else 0.0
            return {
                "vocab": self.vocab, "distinct_tokens": distinct,
                "vocab_coverage": round(coverage, 4),
                "documents": self.documents, "tokens_observed": self.tokens,
                "a": self.a, "informed": self.informed,
                "min_observations": self.min_observations, "version": self.version,
                "weighting": "sif" if self.informed else "uniform (cold start)",
            }
