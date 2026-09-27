# Hybrid Retrieval

```
query → understand → cache? → embed → recall (dense + sparse) → fuse
      → rerank → score → graph boost → diversity → conformal → answer
```

Every stage's latency and every score's components are recorded, so a result can
be *explained* rather than merely returned.

## The two spaces

**Dense** is the pretrained 32000 × 256 embedding table, compiled to ONNX at
first boot and executed locally. It catches paraphrase: *"the coolant is too
low"* finds *"pressure below 1.8 bar is a hard stop"*.

**Sparse** is a SPLADE-shaped, BM25-backed encoder with learned-on-device IDF.
It catches what dense compression destroys: part numbers, error codes, the
things an operator actually types. Impact-pruned to the top 64 terms, because
the tail of a sparse vector costs postings and buys nothing.

Neither is sufficient. `test_sparse_recall_finds_a_rare_token_dense_misses`
exists because a bi-encoder genuinely cannot find `BX-7741-Q`.

## Fusion

Reciprocal rank fusion combines the two *orderings* rather than their scores,
which are not on the same scale and cannot be added. RRF is robust without
per-corpus tuning — the right property for a device with nobody to tune it.
This node weights dense slightly above sparse (0.85) with a constant of 60.

Either the engine does this ([Qdrant at the Centre](Qdrant-at-the-Centre)) or
the interpreter does. The two agree exactly once given the same constant.

## Rerank

A bi-encoder compresses a whole passage into one vector: a long procedure with
one relevant step looks distant from a query about that step. The reranker
scores query and document **token by token** — ColBERT-style MaxSim, where each
query token takes its best match in the document and those maxima are averaged.

It runs on the shortlist only. That is what makes it affordable on a device with
no GPU: the expensive stage sees ten candidates, not ten thousand. Retrieval
score is evidence too, so the reranker *refines* rather than replaces —
`0.75 × maxsim + 0.25 × retrieval`.

## Scoring

Rerank score is one term. The final ordering also weighs recency (exponential
decay, 21-day half-life), salience, confidence, and whether a memory has been
superseded. Every component comes back in the response, so a result that looks
wrong can be argued with.

## What comes after

| Stage | What it adds |
|---|---|
| **Graph boost** | Memories connected to the query's entities, even when their text does not look like the query — the half of recall that embeddings structurally cannot reach |
| **MMR diversity** | Stops five near-identical memories filling a five-slot answer |
| **Conformal prediction** | A calibrated guarantee with realised coverage, not a softmax dressed up as confidence |
| **Contradiction detection** | Flags when retrieved memories disagree with each other |
| **Escalation** | Optional Triton hand-off for genuinely complex queries — declined while offline, and declined for restricted results |

## The index underneath

There is no single best index, so the node measures. `scripts/strategy_bakeoff.py`
forced each strategy onto the same corpus:

| 20,000 points | p50 | recall@10 | build |
|---|---|---|---|
| **flat** | **0.886 ms** | **1.000** | 0 s |
| hnsw | 4.340 ms | 0.773 | 252 s |
| ivf_pq | 550.512 ms | 0.997 | 123 s |

Exhaustive search was 4.9× faster than the graph, exact where the graph lost a
quarter of its recall, and free to build — while the cost model was selecting
the graph from 5,000 points upward. It had timed a graph hop as one vectorised
numpy call, which captures the arithmetic and none of the per-node interpreter
bookkeeping that dominates a real traversal. The hop is now timed with its
bookkeeping (overhead 12.1×) and the crossover solved from the two measurements:
**110,000 points**.

IVF-PQ is not a latency structure here at all. It exists for when RAM, not time,
is the binding constraint, and `choose()` treats that as a separate decision.

## Caching

Two layers, both namespaced by tenant, collection, mode, k and a canonical
digest of the filter spec — because a cache keyed on the query embedding alone
is a cross-tenant leak waiting to happen.

- **Exact**, before the encoder: an identical question skips the model entirely.
- **Semantic**, on the query vector: a near-identical question reuses the answer.

A cached answer carries its confidence and diversity — those are properties of
the answer — but never the stored query's trace or stage timings, which are
narration about work this query did not do.
