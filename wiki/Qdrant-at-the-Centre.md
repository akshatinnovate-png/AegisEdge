# Qdrant at the Centre

Qdrant is not a storage detail here. It is the query engine, and the node is
honest about the one case where it is not.

## The schema every collection gets

| Vector | Type | Why |
|---|---|---|
| `dense` | named, cosine | Semantic recall |
| `lex` | sparse | Rare tokens — part numbers, error codes, the things an operator actually searches for |
| `late` | multivector, `MAX_SIM` comparator, binary-quantized, HNSW disabled | Engine-side ColBERT rerank. Off by default; see the cost below |

The `late` vector's HNSW graph is deliberately disabled: a multivector is only
ever reached through a prefetch, so a graph over it would be built and never
traversed.

## Four stages, one call

The node's hybrid retrieval is dense recall, sparse recall, rank fusion and a
late-interaction rerank. Every one of those is a thing Qdrant does natively, so
all four are expressed as a single `query_points`:

```
prefetch(dense, limit=fetch)  ┐
prefetch(lex,   limit=fetch)  ┴─→ FusionQuery(RRF) ─→ query(late, MaxSim)
```

### Keeping the explanation

Fusing inside the engine returns one ordering, which loses *which half of hybrid
retrieval found a memory* — and that is one of the few things an operator
inspecting a result actually wants to know. It is the difference between "the
wording was close" and "the part number matched".

Asking for the per-space orderings afterwards would cost two more round trips,
which is the cost the single call was supposed to remove. They ride back in the
same batch instead: `query_batch_points` sends the fused query and both
per-space queries as one request. `plan.queries_in_call` is 3; `plan.calls` is 1.

## The bake-off, which said no

`scripts/qdrant_bakeoff.py` runs the same queries down both paths on the same
corpus. At **2,000 points, 64 queries, k=5**, on the embedded client:

| | local index | qdrant engine |
|---|---|---|
| latency p50 | **2.52 ms** | 60.74 ms |
| latency p95 | **3.73 ms** | 76.46 ms |
| dense recall@5 vs exact brute force | 1.000 | 1.000 |

The engine is twenty-four times slower *here*, and that is the expected answer
rather than a disappointment: the embedded client's local mode is a pure-Python
implementation intended for development, while this node's index is a calibrated
HNSW with a quantized cold tier. Against a Qdrant Server the engine is compiled
and the four round trips are real, and the trade flips.

## The disagreement, attributed rather than averaged

The two paths returned the same top-5 only **52%** of the time. On its own that
number is worthless — it could be a recall bug, a scoring bug, or nothing. So
the bake-off decomposes it:

| | agreement |
|---|---|
| dense space alone | **1.000** |
| sparse space alone | **1.000** |
| interpreter's fusion re-run on the engine's RRF constant | **1.000** |
| engine-side pre-filter vs engine-side unfiltered | **1.000** |

Every stage is identical. The whole top-k difference is the fusion constant:
Qdrant's RRF is unweighted with a constant of 1, this node's weights dense above
sparse with a constant of 60. A parameter, not a defect — and the script's
verdict line says so in those words, computed from the table rather than typed
into it.

### It found a bug in our own code

The sparse spaces did not agree at first: **0.717**. The cause was ours. The
encoder's BM25 weights already carry a length normalisation, and
`SparseIndex.search` divided by the document's L2 norm as well — so a long
memory sank for being long, twice. Qdrant's sparse index computes the impact dot
product, which is also the textbook BM25 score. Dropping the second
normalisation moved sparse agreement to **1.000**.

That is a comparison against an independent implementation finding a ranking bug
no self-consistent test could have.

## So the node routes

`aegis/retrieval/routing.py` sends a query to the engine while the engine's own
measured p95 leaves room under the query objective — a quarter of it, because
retrieval is one stage of a request that also embeds, reranks, scores and
serialises — and to the local index when it does not.

- A **Qdrant Server** skips the weighing entirely: one call beats four.
- A **degraded** node takes the faster path; undercutting the SLO ladder while
  it sheds features would defeat what it is defending.
- **One query in thirty-two** goes back to the path that lost, because a path
  that stops being used stops being measured.

Over 24 live queries on a 2,000-point corpus: 4 to the engine (the warm-up),
15 to the local index once its p95 was known, 5 answered from the semantic cache
and never routed at all.

## Two things the engine is never asked

**A filter it cannot express faithfully.** `translate()` maps the node's filter
tree onto Qdrant's and returns `None` — *stay on the local index* — for anything
it cannot express exactly: `CONTAINS` (which reads both substrings and list
membership in these payloads), `EXISTS` (which is not Qdrant's is-null), a
negation inside an OR. A native search that quietly dropped a restriction would
answer a filtered query with unfiltered results. `plan.fell_back` carries the
reason, and the caller reads *that* rather than the empty result list, because
"the engine declined" and "there are no matching memories" are different
answers.

**An allow-set.** When the planner resolves a filter to explicit ids — which is
how tenancy works — the router is not consulted at all. Qdrant filters on
payload; it cannot be handed a set of ids to restrict to. That is correctness,
not latency.

## Late interaction is priced, not enabled

A full ColBERT residual is `tokens × dim` floats: at 128 tokens and 256
dimensions, **128 KB against the ~2 KB a memory costs today**. So the late
vector is pruned to a token budget — keeping the tokens *furthest* from the
document's own centroid, since MaxSim's value is carried by a document's unusual
tokens — binary-quantized, and off by default.

`AEGIS_QDRANT_LATE_TOKENS=32` turns it on, `/api/v1/qdrant` reports the bytes per
memory it costs, and the reranker's existing token vectors supply it so nothing
is embedded twice.

## Migration is deliberate

A node created before this schema holds one unnamed vector per collection.
Nothing is silently rewritten at boot: such a collection is classified `legacy`,
keeps the local-index path, and is left alone. `scripts/qdrant_migrate.py
--apply` moves it — and can only do that honestly because Qdrant is not the only
copy. The write-ahead log and the local index hold every memory, and the
migration hands them back.

If Qdrant holds *more* points for a collection than the node can supply, that
difference is data only Qdrant has, and the migration refuses to touch that
collection unless `--force` says otherwise.

## Payload indexes, reported honestly

The embedded client warns that payload indexes have no effect in local mode.
`/api/v1/qdrant` reports them as `ignored-by-local-mode` rather than as live,
and the console prints `48 PAYLOAD IDX INERT (LOCAL MODE)`. Reporting them as
live would be the one lie this console exists to avoid.
