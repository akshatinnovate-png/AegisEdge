# Architecture

## Shape

```
                      ┌──────────────────────────────────────────┐
   HTTP / WebSocket   │  FastAPI  ·  routers, WS gateway         │
        ───────────▶  ├──────────────────────────────────────────┤
                      │  EdgeNode — composition root             │
                      │    every subsystem built and supervised  │
                      ├───────────────┬──────────────┬───────────┤
                      │  retrieval    │  memory      │  sync     │
                      │  pipeline     │  store       │  engine   │
                      │  + router     │  + Qdrant    │  + egress │
                      ├───────────────┴──────────────┴───────────┤
                      │  inference · policy · learning · renewal │
                      │  chaos · sim · core (clock, bus, SLO)    │
                      └──────────────────────────────────────────┘
                                       │
                          ┌────────────┴────────────┐
                          │  Qdrant (embedded)      │
                          │  dense · lex · late     │
                          └─────────────────────────┘
```

## The modules

| Path | What lives there |
|---|---|
| `aegis/node.py` | Composition root — every subsystem is built and supervised here |
| `aegis/config.py` | All configuration, env-overridable (`AEGIS_*`) |
| `aegis/core/` | HLC clock, event bus, supervisor, metrics, breaker, backoff, token bucket, QoS scheduler, span tracing, tenancy, SLO ladder, swappable ambient environment (virtual clock + seeded RNG) |
| `aegis/memory/` | Schema, WAL, quantizers, RaBitQ cold codes, growable matrices, HNSW, OPQ/IVF-PQ, adaptive index + cost model, filters & payload index, query planner, memmap cold tier, immutable segments + manifest, fsck/scrub/PITR, self-healing repair, bitemporal knowledge graph, **Qdrant hybrid schema and native query path**, compactor, consolidation |
| `aegis/inference/` | ONNX session + execution-provider ladder, micro-batcher, embedder, corpus geometry (whitening), token lexicon, adaptation gate, sparse encoder, reranker, classifier, thermal governor, model registry, Triton client |
| `aegis/retrieval/` | RRF fusion, scoring, namespaced semantic cache, contradiction detection, query understanding, late interaction (MaxSim), conformal prediction, MMR diversity, **latency-aware routing**, pipeline, agent |
| `aegis/sync/` | CRDT op log, Merkle digests, IBLT set reconciliation, vector clocks + causal delivery, P2P gossip mesh, Ed25519 identity + operation signing, wire codec, durable queue, **value-per-byte egress scheduling**, connectivity oracle, transports, conflict arbiter, engine |
| `aegis/sim/` | Deterministic simulation — a virtual world of N peers, weighted fault and Byzantine actions, invariants checked after every step |
| `aegis/learning/` | On-device retrieval adapter, differential privacy, federated secure aggregation |
| `aegis/renewal/` | Freshness sweeps, dual-space migrator, scheduler |
| `aegis/policy/` | Policy engine, redaction vault, hash-chained audit log |
| `aegis/api/` | Routers, schemas, WebSocket gateway |

## Two decisions worth knowing about

**The node measures instead of assuming.** There is no single best index, so at
boot it microbenchmarks the machine it is on, derives the crossover points, and
places each collection on the strategy its size justifies. The same logic
governs which engine answers a query and what the link should carry first.

**Every measurement has a control.** A signed simulation sweep is paired with an
unsigned one. The Qdrant path is paired with the local index. Value-ordered
egress is paired with FIFO. A number with nothing to compare against is not
evidence, and CI enforces this: the build fails if the unsigned control comes
back *clean*.

## Storage layers

| Layer | Holds | Survives |
|---|---|---|
| Write-ahead log | Every mutation, fsynced before ack | Process kill |
| Immutable segments + manifest | Archived history with a checkpoint LSN | Corruption of any one segment |
| Qdrant collections | The vectors and payloads, as the system of record | Restart |
| Local adaptive index | Hot query path, HNSW + quantized cold tier | Rebuilt from the above |

Qdrant is not the only copy, which is precisely what makes a schema migration
safe — see [Qdrant at the Centre](Qdrant-at-the-Centre).
