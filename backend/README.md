# AegisEdge — backend

The edge node. FastAPI on the outside, an offline-first memory and sync
engine on the inside.

```bash
pip install -r requirements.txt
uvicorn aegis.main:app --reload --port 8000     # API + WebSocket
python3 scripts/demo.py                          # full lifecycle, no server
python3 scripts/bench.py                         # index recall + latency on this machine
python3 scripts/stress.py --phases all --out ../testlogs   # the nine-phase battery
python3 scripts/strategy_bakeoff.py              # flat vs HNSW vs IVF-PQ, same corpus
python3 scripts/geometry_eval.py                 # embedding-space sweep with ground truth
python3 -m pytest tests -q                       # 247 tests
```

Open `http://localhost:8000/docs` for the live OpenAPI surface, or point the
frontend at it (it defaults to `http://localhost:8000`).

## Module map

| Path | What lives there |
|---|---|
| `aegis/node.py` | Composition root — every subsystem is built and supervised here |
| `aegis/config.py` | All configuration, env-overridable (`AEGIS_*`) |
| `aegis/core/` | HLC clock, event bus, supervisor, metrics, breaker, backoff, token bucket, QoS scheduler, span tracing, **tenancy**, **SLO ladder** |
| `aegis/memory/` | Schema, WAL, quantizers, **RaBitQ cold codes + columnar codebook**, **growable matrices**, HNSW, OPQ / IVF-PQ, adaptive index + cost model, filters & payload index, query planner, memmap cold tier, **immutable segments + manifest**, **fsck/scrub/PITR**, **self-healing repair**, **bitemporal knowledge graph**, Qdrant Edge adapter, compactor, consolidation |
| `aegis/inference/` | ONNX session + EP ladder, micro-batcher, embedder, **corpus geometry (whitening)**, **token lexicon**, **adaptation gate**, sparse encoder, reranker, classifier, thermal governor, model registry, Triton client |
| `aegis/retrieval/` | RRF fusion, scoring, namespaced semantic cache, contradiction detection, query understanding (BK-tree, expansion, intent), late interaction (MaxSim), **conformal prediction**, **MMR diversity**, pipeline, agent |
| `aegis/sync/` | CRDT op log, Merkle digests, **IBLT set reconciliation**, **vector clocks + causal delivery**, **P2P gossip mesh**, **wire codec**, durable queue, connectivity oracle, transports, conflict arbiter, engine |
| `aegis/learning/` | **On-device retrieval adapter**, **differential privacy**, **federated secure aggregation** |
| `aegis/renewal/` | Freshness sweeps, dual-space migrator, scheduler |
| `aegis/policy/` | Policy engine, redaction vault, hash-chained audit log |
| `aegis/chaos/` | Fault injection |
| `aegis/api/` | Routers, schemas, WebSocket gateway |

## Dependencies are not optional

The node refuses to start without real weights or a real vector store. There is
no fallback encoder and no internal store masquerading as Qdrant.

| Package | Provides |
|---|---|
| `wordllama`, `safetensors` | the pretrained 32000 × 256 token embedding table and its 32k BPE tokenizer, shipped in the wheel — provisioning never touches the network |
| `onnx`, `onnxruntime`, `tokenizers` | the two graphs compiled from those weights at first boot, and the runtime that executes them |
| `qdrant-client` | Qdrant, embedded by default or a server via `AEGIS_QDRANT_URL` |
| `psutil` | platform thermal/battery telemetry; where a platform exposes none, the reading is reported **unavailable** rather than synthesised |

Only `tritonclient[grpc]` is optional, because it needs a reachable GPU server.

On first boot the node compiles `models/embedder.onnx` and
`models/reranker.onnx` from the bundled weights, writes `models/tokenizer.json`
and `models/provenance.json`, and content-addresses both graphs into the
registry. Subsequent boots load the serialized optimized graph.

## Configuration worth knowing

| Variable | Effect |
|---|---|
| `AEGIS_QDRANT_URL` | point the store at a Qdrant Server instead of the embedded instance |
| `AEGIS_CLOUD_URL` | the sync coordinator: a Qdrant URL, or empty for a real embedded Qdrant under `<data_dir>/cloud` |
| `AEGIS_REQUIRE_QDRANT=0` | explicitly allow the internal store (it will say so in `/health`) |
| `AEGIS_REQUIRE_AUTH=1` | enforce API keys and tenant resolution on every route |
| `AEGIS_DATA_DIR` | one node per directory — embedded Qdrant is single-writer |

## Index selection

There is no single best index, so the node measures rather than assumes. At
boot it microbenchmarks this machine, derives the crossover points, and places
each collection on the strategy its size justifies.

It was measuring the wrong thing until `scripts/strategy_bakeoff.py` forced each
strategy onto the same corpus:

| 20,000 points | p50 | recall@10 | build |
|---|---|---|---|
| **flat** | **0.886 ms** | **1.000** | 0 s |
| hnsw | 4.340 ms | 0.773 | 252 s |
| ivf_pq | 550.512 ms | 0.997 | 123 s |

Exhaustive search was 4.9x faster than the graph, exact where the graph lost a
quarter of its recall, and free to build — while the cost model was selecting
the graph from 5,000 points upward. It had timed a graph hop as one vectorised
numpy call, which captures the arithmetic and none of the per-node interpreter
bookkeeping that dominates a real traversal, so the overhead ratio came out
about 1 and the crossover collapsed onto its floor. The hop is now timed with
its bookkeeping (overhead 12.1x) and the crossover solved from the two
measurements: **110,000 points**.

IVF-PQ is not a latency structure here at all — 550 ms against flat's 0.9 ms. It
exists for when RAM, not time, is the binding constraint, and `choose()` treats
that as a separate decision rather than a consequence of corpus size.

Run `scripts/bench.py` or `scripts/strategy_bakeoff.py` to reproduce those
numbers, or disagree with them.

Two things the calibration decides for itself, because they are properties of
the *data*, not of the algorithm:

- **nprobe** — a true neighbour sitting in an unprobed IVF cell cannot be
  recovered at any rescore depth. Clustered corpora need ~24% of cells probed;
  uniformly distributed ones need ~90%, because IVF has no structure to
  exploit there.
- **rescore depth** — quantization error routinely exceeds the gaps between
  near neighbours, so the right answers sit deep in the approximate ordering
  even at 0.995 rank correlation. The index measures the depth its corpus
  needs for the target recall instead of guessing `4k`.

## Survival properties

The `tests/test_survival.py` suite asserts invariants under simultaneous fault
storms rather than happy paths:

- a query is always answered or fails loudly — never hangs;
- every accepted write is resident, durable, or reported lost by identifier;
- tenant isolation holds at **every** degradation level;
- restricted memories never leave under packet loss, clock skew or partition;
- repeated restarts converge to one state.

## Things worth knowing

- **`divergent ranges` rarely reaches zero, and that is correct.** Points the
  policy marks `local_only` are never offered to the coordinator, so those
  Merkle buckets legitimately differ forever. Convergence means *everything
  allowed to sync has synced*, not that the two sides are identical.
- **`loopback://cloud` is not a mock.** It keeps its own op log and Merkle
  tree, really diverges and really converges. Point `AEGIS_CLOUD_URL` at an
  HTTP coordinator to use `HttpCloud` instead.
- **The WAL is the system of record.** Nothing becomes searchable before it is
  recoverable.
- **The node ships empty.** There are no seeded demo memories; the demo script
  and the test suite ingest their own corpus through the same path a deployed
  device uses.
- **Embedded Qdrant is single-writer.** Two nodes on one `AEGIS_DATA_DIR` is
  not a supported configuration, and the error says so in those words. A
  restart must close its handles first — `EdgeNode.close()` is part of the
  restart contract, not a tidiness nicety.
- **The cold tier really leaves RAM.** Cold vectors are evicted to a memmapped
  file; only 1-bit codes stay resident, and rescoring pages back just the
  shortlist. Keeping a full-precision copy "for rescoring" would make the
  compression ratio a slide rather than a fact.
- **Federated learning states what it needs.** Gaussian DP noise is
  per-coordinate, so one device's update is mostly noise by construction. Each
  round reports its SNR and the cohort size that epsilon would actually
  require (hundreds of thousands, at eps=2 over 12k parameters). DP can be
  switched off for a small fleet — that is an audited choice, and secure
  aggregation still hides the individual update either way.
- **A corrupt segment is not automatically lost data.** Most of those memories
  are still in RAM; the durable copy is gone, not the memory. The archive
  watermark rewinds and a fresh segment is sealed. Reporting those as
  "recovered from a peer" would overstate what the repair did.
- **The semantic cache is keyed on the whole scope, not just the tenant.** Two
  tenants asking the same question produce the same embedding — a cache keyed
  on the vector alone is a cross-tenant leak, and was one until a test caught
  it. Keying on the tenant alone is the same bug wearing a different hat: a
  query run over all collections was then served verbatim for
  `collection="procedural"`, five hits from a collection holding nothing.
  Collection, mode and `k` all change what a correct answer *is*, so they are
  all in the key.
- **An inferred narrowing can never empty the results.** Query understanding
  may guess a collection from intent; if the guess matches nothing, the query
  is retried at the scope the caller actually asked for and the result records
  what was dropped. Only what that layer added is backed out — an explicit
  filter is obeyed even when it matches nothing, and tenant isolation is never
  relaxed, because an empty visible set is a boundary rather than a bad guess.
- **The PII classifier is on the ingest path, so its worst case is a weapon.**
  One unanchored regex made a single 1 MB write take 79 minutes at 100% CPU.
  Patterns are anchored and the scan runs in overlapping windows, so no pattern
  is ever handed an unbounded string. Truncating the scan was rejected: a
  secret at offset two megabytes must not escape classification because
  scanning it was inconvenient.
- **Adaptations are armed by evidence, not by citation.** `AdaptationGate`
  scores a candidate embedding space on a paraphrase task built from the node's
  own memories and adopts it only when a paired bootstrap puts the 95% interval
  clear of zero. SIF pooling — a heavily cited result — measured *worse* on
  this model at every setting and ships off because of it.
- **Causal ordering applies to the rumour path, not to bulk transfer.** CRDT
  operations are commutative, so anti-entropy applies a set directly; vector
  clocks guard the streaming path, where a supersede can outrun what it
  supersedes.
