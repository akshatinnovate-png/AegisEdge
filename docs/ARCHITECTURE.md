# Architecture — every part, and why that mechanism

*Part of [AegisEdge](../README.md). System shape, the feature plan, the API surface, and a module-by-module account of the node.*

---

## 5. System shape

```
                          ┌───────────────────────────────────────────┐
  CONTROL PLANE           │  Tenants · API keys + scopes · Quotas     │
                          │  Policy (hot-reload) · Hash-chained audit │
                          │  SLO objectives · Degradation ladder      │
                          └────────────────────┬──────────────────────┘
                                               │ governs every plane below
╔══════════════════════════════════════════════▼══════════════════════════════════════════════╗
║  EDGE NODE — offline-capable, multi-tenant, self-healing                                     ║
║                                                                                              ║
║  INGEST PLANE                                                                                ║
║   ingest ─▶ quota gate ─▶ sensitivity classifier ─▶ policy engine ─▶ redaction vault          ║
║                 │                    │                    │                                  ║
║                 ▼                    ▼                    ▼                                  ║
║        ONNX embedder          entity + relation     sync class decision                      ║
║        (EP ladder,            extraction            (local / redacted /                      ║
║         µ-batch, int8)              │                metadata / full)                        ║
║                 │                   ▼                      │                                 ║
║                 │        ╔══════════════════════╗          │                                 ║
║                 │        ║ BITEMPORAL KNOWLEDGE ║          │                                 ║
║                 │        ║ GRAPH                ║          │                                 ║
║                 │        ║ valid-time ⊥ tx-time ║          │                                 ║
║                 │        ║ retract ≠ delete     ║          │                                 ║
║                 │        ╚═══════════╤══════════╝          │                                 ║
║                 ▼                    │                     ▼                                 ║
║  STORAGE PLANE  │                    │        write-ahead log (CRC, torn-tail safe)          ║
║   ┌─────────────▼────────────────────┼───────────────────────────┐        │                  ║
║   │ ADAPTIVE INDEX  (cost model calibrated on this silicon)      │        ▼                  ║
║   │   FLAT BLAS  ⟷  HNSW graph  ⟷  IVF-PQ + OPQ (64× smaller)    │  immutable segments       ║
║   │   sparse postings (BM25/SPLADE) · payload index + stats      │  content-addressed        ║
║   │   HOT full ▸ WARM int8 ▸ COLD 1-bit + vectors on memmap      │  manifest (atomic swap)   ║
║   └─────────────┬────────────────────────────────────────────────┘  fsck · scrub · PITR      ║
║                 │                                                        │                   ║
║  RETRIEVAL PLANE▼                                                        ▼                   ║
║   understand (BK-tree repair · expansion · intent · units/time)    self-healing repair        ║
║        ▼                                                          (peer ▸ cloud ▸ report)    ║
║   cost-based planner ─ pre-filter │ post-filter │ scan │ provably empty                       ║
║        ▼                                                                                     ║
║   dense ⊕ sparse ─▶ RRF ─▶ cross-encoder ─▶ MaxSim late interaction ─▶ on-device adapter      ║
║        ▼                                                                                     ║
║   graph boost (spreading activation) ─▶ MMR diversity ─▶ CONFORMAL prediction set             ║
║        ▼                                                          (coverage guarantee /       ║
║   agentic reasoner: plan ▸ retrieve ▸ verify ▸ cite                honest abstention)          ║
║                                                                                              ║
║  RUNTIME PLANE                                                                               ║
║   QoS scheduler (interactive ▸ sync ▸ maintenance ▸ renewal, deadlines, shedding)             ║
║   supervisor · span tracing · metrics · thermal/power governor · chaos injection              ║
║   event bus ─▶ multiplexed WebSocket gateway                                                 ║
╚═══════════════╤══════════════════════════════════╤═══════════════════════════════════════════╝
                │                                  │
     ┌──────────▼───────────┐         ┌────────────▼─────────────────────────────┐
     │  PEER MESH           │         │  intermittent, hostile, lossy uplink      │
     │  IBLT reconciliation │         │  QUIC 0-RTT · circuit breaker · hedging   │
     │  vector clocks +     │         │  token-bucket budget · resumable cursor   │
     │  causal delivery     │         └────────────┬─────────────────────────────┘
     │  gossip + rumour     │                      ▼
     │  policy at the edge  │      ╔═══════════════════════════════════════════════╗
     └──────────┬───────────┘      ║  CLOUD TIER                                   ║
                │                  ║  Sync coordinator · Qdrant Server (canonical) ║
                └──────────────────║  Triton (ensembles, dynamic batching)         ║
       device ⇄ device, no cloud   ║  Renewal orchestrator · Fleet registry        ║
                                   ║  Conflict arbiter · Federated aggregator      ║
                                   ║  (secure aggregation — never sees an update)  ║
                                   ╚═══════════════════════════════════════════════╝
```

Four properties the diagram is making claims about, each checked by a test:

1. **Nothing crosses a boundary it was not allowed to** — not to the cloud, not
   to a peer, not to another tenant, not through the cache, at any degradation
   level.
2. **Every accepted write is either resident, durable, or reported lost** — by
   identifier, never silently.
3. **A query is always answered or fails loudly** — the ladder sheds stages
   rather than letting latency run away.
4. **Every claim about confidence is calibrated** — coverage is measured
   against realised outcomes, not asserted.

---

## 6. Backend feature plan

### 6.1 Memory core — Qdrant Edge

| Feature | Detail |
|---|---|
| Qdrant is required | Qdrant is a **hard dependency**, embedded by default and reported verbatim in `/health` as `qdrant-local` or `qdrant-server`. A node that silently falls back to an internal store while claiming to be Qdrant-backed is telling its operator something untrue about where their data lives. `AEGIS_REQUIRE_QDRANT=0` allows the internal store *explicitly*. |
| Two paths, verified to agree | Qdrant owns persistence and the collection API; the local adaptive index serves hot queries. `verify_agreement()` proves the two return the same neighbours instead of asking anyone to assume it — **100% overlap** measured. |
| Same code, server or embedded | `AEGIS_QDRANT_URL=http://host:6333` moves the same adapter onto a Qdrant Server, where the engine is Rust and the trade-off flips. |
| Single-writer, stated | Embedded Qdrant takes an exclusive lock on its directory. A second node on the same `AEGIS_DATA_DIR` gets a message that says so, not a `BlockingIOError` from three libraries down. |
| Hybrid vectors | Every point carries a **dense** vector (`bge-small-en-v1.5`, 384d) *and* a **sparse** vector (SPLADE-mini / BM25 fallback) so lexical rare-token matches survive offline. |
| Named vector spaces | Multi-vector points: `text`, `vision`, `audio`, `fused` — one point can be retrieved through any modality. |
| Scalar + binary quantization | INT8 scalar quantization on the hot tier, **binary quantization** on the cold tier for a 32× memory drop, with rescoring from the full vectors on disk. |
| Payload indexing | Keyed indexes on `ts`, `geo`, `device_id`, `sensitivity`, `model_version`, `ttl` to keep filtered search pre-filtered rather than post-filtered. |
| Memory tiering | **Hot** (mmap'd, full precision) → **Warm** (INT8) → **Cold** (binary + on-disk payload) → **Evicted** (sync'd to cloud, tombstone kept). A background compactor moves points across tiers on an access-recency × salience score. |
| Snapshotting | Periodic consistent snapshots to local object storage; a corrupted segment restores from the last snapshot + WAL replay instead of a full resync. |

### 6.2 Local inference — real weights, real ONNX

The node runs **pretrained weights**, not a stand-in, and it will not start
without them. There is no synthetic encoder to fall back to, because a device
that silently downgrades to a toy model produces confident nonsense and nobody
is there to catch it.

| Feature | Detail |
|---|---|
| Pretrained model | A 32000 × 256 token embedding table distilled from Llama-3 (`wordllama` l2_supercat), with its real 32k BPE tokenizer. Both ship **inside the Python distribution** — provisioning needs no model hub, no download, no network. A device in a tunnel cannot fetch weights. |
| Compiled locally | Two ONNX graphs are built from those weights at first boot: the **embedder** (gather → masked mean pool → L2 normalise) and the **reranker** (normalised per-token vectors for MaxSim). Compiled once, content-addressed, cached. |
| Measured quality | Paraphrase similarity **0.77–0.96**, unrelated **0.10–0.32**. Embedding latency **0.04 ms/query** on CPU. |
| Execution-provider ladder | Probes and takes the best of **TensorRT → CUDA → ROCm → OpenVINO → CoreML → NNAPI → QNN → XNNPACK → CPU**. The provider actually in use is reported in `/health`. |
| Graph optimizations | `ORT_ENABLE_ALL` with ahead-of-time serialization, so a cold start is a load rather than a re-optimization — the difference between ready in 200 ms and ready in ten seconds on a device that reboots with the vehicle. |
| Dynamic micro-batching | An 8 ms coalescing window batches concurrent embed calls into one session run. |
| Model registry | Content-addressed (`sha256`) with digest verification before a graph is loaded — checking something real, not a placeholder. |
| Governor, honestly | The graph is a gather plus a pooling reduction: there is **no separate int8 artefact to swap to**. What the governor actually controls is the **token budget** and batching window, which is where the cost lives. Claiming a quantized hot-swap would be a dashboard lie. |

### 6.3 Reranking — late interaction, not lexical overlap

A bi-encoder compresses a whole passage into one vector, so a long procedure
with one relevant step looks distant from a query about that step. The reranker
scores **token by token** — ColBERT-style MaxSim, each query token taking its
best match in the document — on the same pretrained space, through its own ONNX
graph, on the shortlist only.

It is strong enough to overturn a misleading retrieval score: in the test
suite a document with retrieval score 0.9 is correctly demoted below one
scoring 0.2. Query→document token alignments are returned as evidence.

### 6.4 Cloud inference — NVIDIA Triton

| Feature | Detail |
|---|---|
| Heavy tier | Large rerankers, VLM captioning, long-context summarization and the "deep reasoning" path run on **Triton** — reached only when the link is up and the policy engine allows it. |
| Ensemble models | Triton **ensemble** pipelines chain `tokenize → embed → rerank` server-side so one gRPC call replaces three round-trips over a bad link. |
| Dynamic batching | Triton dynamic batcher (`max_queue_delay_microseconds`) plus multiple model instances per GPU for fleet-wide throughput. |
| gRPC streaming | Bi-directional streaming inference so partial results reach the device as they are produced — a dropped link loses the tail, not the whole response. |
| Escalation policy | Local ONNX answers first and always. Triton is consulted only when local confidence < τ, the query is flagged complex, and RTT/jitter budget is met. Every escalation is logged with the reason, visible in the UI. |
| No stand-in | An earlier version returned random scores when no server was reachable, so the code path stayed "measurable". That is a lie with a latency histogram attached — it would have reordered real results using noise. Without a live endpoint the escalation is **declined**; with one configured but unreachable it raises so the breaker opens. |
| Model parity guard | Triton and ONNX embedders are version-locked. A mismatch triggers **renewal** (§6.12 below), never a silent mixed-embedding-space corruption. |

### 6.4 Approximate nearest neighbour — measured, not assumed

| Feature | Detail |
|---|---|
| HNSW | Full hierarchical graph with **heuristic neighbour selection** (naive top-M builds hubs and collapses recall on clustered data), bidirectional pruning, soft deletes with graph repair, entry-point demotion. **0.993 recall@10** measured. |
| OPQ + IVF-PQ | Product quantization on IVF residuals with a learned **OPQ rotation** — plain PQ slices by position, which assumes variance is already evenly spread; embeddings are nothing like that. **48-64x compression** at 0.993 recall. |
| Self-calibration | `nprobe` and rescore depth are *measured per corpus*, not guessed: clustered data needs ~24% of cells probed, uniform data ~90%. The index reports both. |
| Cost model | The node microbenchmarks its own silicon at boot and derives the flat→HNSW→IVF-PQ crossovers from it. A collection migrates strategy as it grows; migrations are logged. |
| Cold tier on disk | Cold vectors are evicted to a **memmap**; only 1-bit codes stay resident and rescoring pages in the shortlist alone. |

### 6.5 Query planning

| Feature | Detail |
|---|---|
| Payload index | Keyword postings plus sorted numeric arrays, carrying cardinality statistics so selectivity is *estimated* before anything executes. |
| Cost-based planner | Chooses pre-filter (resolve ids, scan that subset exactly), post-filter (ANN first, over-fetching by inverse selectivity) or full scan — and explains the choice in the response. |
| Provable emptiness | A filter that matches nothing does **zero** vector work. |
| Cross-collection | A `*` search aggregates per-collection plans instead of letting an empty one veto the query. |

### 6.6 Query understanding — local, in under 3 ms

| Feature | Detail |
|---|---|
| BK-tree spelling repair | Metric tree over the *corpus* vocabulary, so "colent presure" becomes "coolant pressure" — domain terms a generic dictionary would never hold. Common English words are protected from over-eager correction. |
| Co-occurrence expansion | PMI-style expansion learned from ingested text; no embedding round trip. |
| Unit & temporal normalisation | `4.2mm/s` is normalised; "from the last 2 hours" becomes a payload filter the planner can use. |
| Intent routing | Procedural / sensor / episodic / semantic routing from the query shape. |

### 6.7 Retrieval & reasoning

- **Hybrid fusion** — dense + sparse candidates merged with **Reciprocal Rank Fusion**, then cross-encoder reranked on-device.
- **Late interaction** — ColBERT-style **MaxSim** over per-token vectors (int8-quantized) on the shortlist only, with query-term → document-term alignments returned as evidence.
- **On-device adapter** — a rank-16 low-rank adapter (~48 KB) trained from real feedback by contrastive updates, so the node learns *this* site's vocabulary without a fine-tune.
- **Span tracing** — every query emits a nested span tree and names its own hotspot.
- **Temporal decay + salience** — final score = `α·similarity + β·recency_decay + γ·access_frequency + δ·pinned` so stale memories sink without being deleted.
- **Contradiction detection** — an NLI head flags memories that contradict newer ones; the loser is *superseded*, not erased, and the chain stays inspectable.
- **Memory consolidation** — a nightly (or idle-triggered) job clusters near-duplicate episodic points and distills them into a single semantic point, with provenance links to the originals. Local memory stops growing linearly with uptime.
- **Agentic loop** — plan → retrieve → (optionally escalate to Triton) → verify → answer, with every step emitted on the telemetry bus so the frontend can render the node's actual reasoning trace.
- **Query cache** — semantic cache keyed on embedding proximity; a near-identical question answers from cache in <1 ms.

### 6.8 Sync engine — edge ⇄ cloud

| Feature | Detail |
|---|---|
| CRDT log | Every mutation is an **LWW-Element-Set / OR-Set** operation with a hybrid logical clock (`HLC`), so two partitioned devices converge without a coordinator. |
| Delta sync | Devices exchange **Merkle-tree range digests** first; only divergent ranges transfer. A 400k-point collection reconciles in kilobytes. |
| Bidirectional | Edge→cloud pushes new local knowledge; cloud→edge pulls fleet knowledge relevant to *this* device's geo/role/task profile (not the whole corpus). |
| Selective sync | The **policy engine** (§6.18 below) decides per-point: `local_only`, `sync_metadata_only`, `sync_full`, `sync_after_redaction`. |
| Conflict arbiter | Resolution ladder: HLC → device trust score → semantic merge (both retained, linked as variants) → human review queue surfaced in the UI. |
| Backpressure | Sync respects a token-bucket bandwidth budget and yields to foreground queries — sync never makes the device feel slow. |
| Resumable transfer | Chunked, checksummed, offset-resumable. A link that dies at 93% resumes at 93%. |
| Tombstones + GC | Deletes propagate as tombstones with a grace window, then are garbage collected fleet-wide. |

### 6.9 Peer-to-peer mesh — no cloud involved

Two robots in a tunnel are ten metres apart and both blind under a
cloud-centric design. AegisEdge lets them reconcile directly.

| Feature | Detail |
|---|---|
| IBLT set reconciliation | Invertible Bloom Lookup Tables sized by the *difference*, not the corpus: two 5000-op devices reconcile a 15-op difference in **6 KB**, in one exchange. Cells XOR the key itself, so neither side needs a dictionary of the other's keys. |
| Honest failure | If the difference outruns the table, decoding reports incomplete and the round retries wider rather than acting on a partial answer. |
| Vector clocks + causal delivery | The rumour path holds an operation until its declared predecessors arrive, so a supersede never lands before what it supersedes. Bulk anti-entropy applies directly — CRDT ops are commutative. |
| Rumour mongering | New facts push to a few random peers immediately and stop when they come back as duplicates: log-round propagation, no broadcast storm. |
| Policy at the peer boundary | A peer is egress. Restricted memories are withheld from peers *and from relays*, and withheld ops never enter the causal sequence, so they cannot stall it. |
| Wire codec | Delta encoding + int8 vectors + zlib: **17.7x** smaller frames at 0.99998 vector fidelity. |

### 6.10 On-device learning

| Feature | Detail |
|---|---|
| Retrieval adapter | Rank-16 projections over the embedding space, trained by contrastive hinge from click feedback. ~48 KB, microseconds per example. |
| Differential privacy | Clipped updates plus calibrated Gaussian noise, with a tracked (ε, δ) budget the node refuses to overspend. |
| Secure aggregation | Pairwise masks cancel exactly in the sum, so the coordinator sees the average and never an individual update. Rounds with too many dropouts are abandoned rather than corrupted. |
| Honest utility reporting | Each round reports its SNR and the cohort size that epsilon would actually need. DP is switchable for small fleets — an audited choice, not a silent one. |

### 6.11 QoS scheduling

Background work is not optional, but a waiting person outranks a re-embedding
batch. Work is admitted into priority lanes (interactive / sync / maintenance /
renewal) with deadlines; stale background jobs are **shed** rather than run
late, and admission control rejects work the node cannot finish instead of
missing every deadline at once.

### 6.12 Data renewal

Memory rots. Renewal is a first-class subsystem, not a cron job.

- **Freshness scoring** — every point carries `ttl`, `confidence`, `last_verified_at`. A decay function marks points `stale` before they mislead anyone.
- **Re-embedding on model upgrade** — when the embedder version bumps, a **dual-space migration** begins: both spaces are queried and results fused while a background job re-embeds the corpus in priority order (hot tier first). Zero read downtime, zero mixed-space corruption.
- **Progressive renewal** — re-embedding is checkpointed and interruptible; power loss resumes from the last checkpoint.
- **Source revalidation** — points linked to an external source are re-fetched and diffed when connectivity allows; changed sources supersede the old memory and keep the audit chain.
- **Compaction & decay** — expired points drop to cold, then to tombstone. Pinned and high-salience points are exempt.
- **Shadow evaluation** — before a renewed model is promoted, a golden query set is replayed against both spaces; promotion is blocked on recall regression.

### 6.13 Bitemporal knowledge graph

Vector search answers "what looks like this query". It cannot answer "which
bearings on line 2 were replaced after the torque fault" — that is structural —
and it cannot answer "what did we believe on Tuesday", because embeddings have
no notion of belief over time.

| Feature | Detail |
|---|---|
| Two time axes | **Valid time** (when the fact was true) and **transaction time** (when this node believed it) are kept separately. "What did we know on Tuesday about Monday's state" becomes a query rather than an archaeology project. |
| Retract ≠ delete | Withdrawing a belief writes to the transaction axis. The fact stays inspectable, so an incident review can see what the node used to think and when it stopped. |
| On-device extraction | Rule-based entity and relation extraction in microseconds — part codes, bays, assets, metrics, operators, thresholds. Not an LLM, and it does not pretend to be. |
| Corroboration, not duplication | The same relation seen twice merges provenance and raises confidence toward — never past — certainty. |
| Multi-hop reasoning | Bounded BFS over the graph *as it was believed* at any instant, with per-path confidence. |
| Graph-boosted retrieval | Spreading activation from the query's entities surfaces memories that are structurally related but textually dissimilar — the half of recall embeddings cannot reach. |

### 6.14 Calibrated confidence and abstention

A similarity score is not a probability; it depends on the encoder, the corpus
and whichever quantized variant the thermal governor swapped in ten minutes
ago. On an edge device a confident wrong answer is worse than no answer,
because nobody is watching to catch it.

Split conformal prediction gives a **distribution-free coverage guarantee**:
the returned set contains the correct answer at least (1−α) of the time,
whatever the score distribution looks like. Measured empirical coverage in the
test suite: **0.93 against a 0.90 target**. Consequences:

- the node can **abstain** with a stated error rate instead of a hunch;
- set size becomes a measured signal of ambiguity;
- realised coverage is audited continuously, so a drifting encoder shows up as
  **drifting coverage** rather than silent degradation.

Calibration samples come from the feedback the node already collects, and
nonconformity is scored on the *gap* to the best candidate, so recalibration
is not needed every time the encoder's absolute scale moves.

### 6.15 Durability engineering

A WAL recovers the last state. It does not protect against a half-written
snapshot, a manifest pointing at an unfsynced segment, or silent bit rot in a
file nobody has read for six months.

| Feature | Detail |
|---|---|
| Immutable, content-addressed segments | A segment is named by the hash of its bytes, so a corrupted segment cannot masquerade as a good one. Per-record CRC, per-segment digest footer. |
| Crash-safe manifest | Temp file → fsync → atomic rename → fsync(dir), with a retained previous manifest. A crash yields the old manifest or the new one, never a blend. |
| fsck + background scrub | Bit rot is found by reading data nobody asked for. Damaged segments are **quarantined for forensics**, never silently deleted. |
| Self-healing | Lost memories are recovered from **peers first** (reachable when the uplink is not, and free), then cloud. What neither can supply is reported by identifier — the loss is auditable, not invisible. |
| Lost data vs lost copy | A corrupt segment whose memories are still resident is not lost data: the archive watermark rewinds and a fresh durable copy is written. Counting those as "recovered from a peer" would be a flattering lie. |
| Point-in-time restore | Generations are retained; restore drops everything after a chosen one. |
| Format versioning | An on-disk format newer than the build refuses to open rather than corrupting it. |

### 6.16 Multi-tenancy

One device often serves an OEM, the line operator and a maintenance
contractor. Tenancy as a payload field is not isolation — one forgotten filter
and the contractor reads the operator's incidents.

- Isolation is **structural**: the store refuses cross-tenant reads rather than
  trusting each call site to remember.
- The semantic cache is **namespaced**, because two tenants asking the same
  question produce the same embedding. (This was a real bug, found and fixed;
  there is now a regression test named after it.)
- Quotas are enforced **before** work is done — points, bytes, ingest rate and
  QPS — so one tenant cannot deny service to the others.
- API keys are stored hashed, compared in constant time, scoped
  (read/write/admin/sync/learn), and revocable.

### 6.17 Survival: the degradation ladder

Most systems degrade by getting slower until something times out, which fails
every request instead of protecting most of them. AegisEdge runs an explicit
ladder driven by error-budget burn rate, shedding the most expensive stage
still enabled:

| Rung | What stops |
|---|---|
| `FULL` | nothing |
| `ECONOMISE` | cloud escalation, wide fetches, graph boost |
| `TRIM` | late interaction, adapter rescoring, query rewriting |
| `ESSENTIAL` | cross-encoder rerank, diversity — fusion only |
| `SURVIVAL` | dense retrieval only; background work suspended |

Rungs are entered on sustained burn and left with hysteresis, and every
transition names what was disabled and why — a silently degraded system is
indistinguishable from a broken one.

### 6.18 Policy & privacy engine

- On-device **PII / sensitivity classifier** (ONNX) tags every chunk `public | internal | sensitive | restricted` before it is ever written.
- Declarative policy (`YAML`, hot-reloadable): *restricted never leaves the device; sensitive syncs only redacted; public syncs freely.*
- **Redaction pipeline** — named-entity masking with a reversible local-only vault so the device can still resolve what the cloud can never see.
- Encryption at rest (AES-256-GCM, key in TPM/Secure Enclave/keyring), mTLS in flight, per-device identity certificates.
- **Tamper-evident audit log** — hash-chained, append-only record of every read, sync and escalation.

### 6.19 Instantaneous reconnection

The part most projects hand-wave. Reconnection is sub-second and stateful.

- **Connectivity oracle** — active probes + OS network-change hooks + RTT/jitter/loss EWMA classify the link as `OFFLINE / DEGRADED / METERED / HEALTHY`. State changes fire in milliseconds, not on the next poll tick.
- **Pre-warmed transport** — QUIC/HTTP3 with **0-RTT session resumption** and a warm connection pool; reconnect skips the full handshake.
- **Session continuation tokens** — the server remembers the device's sync cursor, so resumption is `"continue from op 84,213"`, not a re-handshake.
- **Operation queue** — every action taken offline is durably queued, idempotency-keyed, and replayed in causal order on reconnect. Nothing is lost, nothing is applied twice.
- **Hedged requests** — during `DEGRADED`, duplicate requests are raced across paths and the first response wins.
- **Reconnect storm control** — decorrelated jitter backoff + fleet-wide token bucket so 10,000 devices returning at once don't DDoS the coordinator.
- **Circuit breaker** — per-endpoint breakers trip fast and half-open probe, so a sick cloud endpoint never stalls the local path.
- **Optimistic UI contract** — the WebSocket pushes `link_state` transitions so the frontend flips between LOCAL and FUSED modes the moment the link moves.

### 6.20 Realtime & transport layer

- **WebSocket multiplex** — one socket, logical channels (`telemetry`, `sync`, `search`, `reasoning_trace`, `alerts`), heartbeats with server-side liveness detection.
- **Event bus** — internal pub/sub; every subsystem emits structured events which the gateway fans out to subscribed UIs.
- **Server-Sent Events fallback** for locked-down networks; **gRPC** for device↔cloud; **REST** for control plane.
- **Backpressure-aware streaming** — slow consumers get sampled, not buffered to death.

### 6.21 Reliability & operations

- **Supervisor** with per-subsystem health, restart budgets and crash-loop detection.
- **WAL + crash recovery** — an unclean shutdown replays the write-ahead log; a half-written batch is never half-visible.
- **Chaos hooks** — inject link loss, packet loss, clock skew, disk-full and process kill from an admin endpoint. The demo can *prove* resilience live, on stage.
- **Observability** — OpenTelemetry traces spanning `edge → link → Triton → back`, Prometheus metrics, structured logs; p50/p95/p99 exposed to the UI.
- **Benchmark harness** — reproducible recall@k, latency and sync-convergence numbers, checked in.

---

## 7. Planned stack

| Layer | Choice |
|---|---|
| Edge runtime | Python 3.11 · FastAPI · Uvicorn · asyncio · NumPy |
| Vector memory | **Qdrant Edge** (embedded) |
| Local inference | **ONNX Runtime** (+ TensorRT/OpenVINO/CoreML/NNAPI EPs) |
| Cloud inference | **NVIDIA Triton Inference Server** (gRPC, ensembles, dynamic batching) |
| Cloud memory | Qdrant Server |
| Transport | WebSocket · QUIC/HTTP3 · gRPC · REST |
| Local durability | SQLite (WAL) for the op log · object store for snapshots |
| Telemetry | OpenTelemetry · Prometheus |
| Frontend | Vanilla HTML/CSS/JS, zero build step (§11) |

---

## 8. API surface (frontend contract)

The frontend in `frontend/` already speaks this contract and degrades to a
local simulation when the backend is absent.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/health` | Node liveness, ONNX execution provider, uptime |
| `GET` | `/api/v1/node/state` | Link state, mode (`LOCAL`/`FUSED`), thermal/power, model versions |
| `GET` | `/api/v1/memory/stats` | Point counts per tier & collection, quantization, disk, renewal progress |
| `POST` | `/api/v1/search` | `{query, k, filters, mode}` → hybrid results + latency breakdown + trace |
| `GET` | `/api/v1/sync/status` | Cursor, pending ops, divergence, last convergence, conflicts |
| `POST` | `/api/v1/sync/trigger` | Force a reconciliation pass |
| `POST` | `/api/v1/memory/ingest` | Push a memory through the ingest pipeline |
| `GET` | `/api/v1/renewal/status` | Re-embedding progress, dual-space migration state |
| `POST` | `/api/v1/chaos/{fault}` | Inject a fault (demo/testing only) |
| `GET` | `/api/v1/index` | Index strategies, calibration, planner statistics |
| `GET` | `/api/v1/traces` | Recent query span trees and their hotspots |
| `GET` | `/api/v1/scheduler` | QoS lane depths, deadline misses, pressure |
| `POST` | `/api/v1/learning/feedback` | Teach the on-device adapter from a real choice |
| `POST` | `/api/v1/learning/round` | One secure-aggregation round |
| `GET` | `/api/v1/mesh/status` · `POST /mesh/round` | Peer mesh membership and anti-entropy. Carries this device's public key, the devices it has learned keys for, and its running counts of operations verified, refused as forged, and refused by policy on arrival |
| `GET` | `/api/v1/graph/stats` · `/entities` · `POST /paths` | Knowledge graph structure and multi-hop reasoning |
| `GET` | `/api/v1/graph/as-of` · `/diff` | Time travel: what was believed, and what changed |
| `GET/POST` | `/api/v1/integrity/*` | fsck, scrub, archive, generations, point-in-time restore |
| `GET/POST` | `/api/v1/slo` · `/slo/override` | Error budget, degradation level, manual pin |
| `GET/POST` | `/api/v1/tenants/*` | Tenants, scoped API keys, quotas |
| `GET` | `/api/v1/learning/confidence` | Conformal calibration and realised coverage |
| `POST` | `/api/v1/ask` | Agentic answer: plan → retrieve → verify → cited answer |
| `GET` | `/api/v1/audit` | Hash-chained audit entries + chain verification |
| `GET` | `/api/v1/metrics` | Prometheus exposition (`/metrics/json` for the raw snapshot) |
| `GET` | `/api/v1/sync/conflicts` | Conflict records and the human review queue |
| `POST` | `/api/v1/mesh/exchange` | The receiving half of the mesh, when the peer is another device |
| `POST` | `/api/v1/mesh/offline` | Pull this device's radio, or put it back |
| `GET` | `/api/v1/integrity/invariants` · `POST /invariants/check` | What the node has asserted about itself while running: the simulator's seven invariants, per-invariant counters, the scope each local check can actually establish, and any findings |
| `POST` | `/api/v1/mesh/attack` | Mount one of four operations — honest, tampered, impersonated, unsigned — against this node through the same handler a peer reaches, and report what it did with it. The honest case is the control |
| `POST` | `/api/v1/chaos/kill` | Send this node SIGKILL; refused without a supervisor |
| `GET` | `/api/v1/integrity/recovery` | What the last boot recovered, and what it could not |
| `GET` | `/api/v1/energy` | Joules per operation, and answers per 1% of battery |
| `GET` | `/api/v1/provenance` · `POST /provenance/verify` | The Merkle receipt, and a file-by-file comparison |
| `GET` | `/api/v1/space` | The embedding space: pooling, fitted geometry, lexicon, gate history |
| `GET` | `/api/v1/space/anisotropy` | Measured conditioning of the stored vectors |
| `POST` | `/api/v1/space/evaluate` | Score candidate spaces against the shipped one — measures, never arms |
| `POST` | `/api/v1/space/arm` | Switch a fitted transform on; refused while any vector is still in the old space |
| `WS` | `/api/v1/stream` | Multiplexed live telemetry, sync events, reasoning traces |

Full surface at `/docs` once the node is running.

---

## 9. Nothing is simulated

Everything on this page is produced by real components. The things that would
normally be faked in a hackathon build, and what they are here:

| Would usually be | Here |
|---|---|
| A hashed or random "embedder" | Pretrained Llama-3-distilled token embeddings, compiled to ONNX, running on ONNX Runtime |
| A lexical-overlap "reranker" | ColBERT-style MaxSim over the same pretrained space, in its own ONNX graph |
| An in-memory dict pretending to be a vector DB | Qdrant, embedded by default, with the two search paths verified to agree |
| A dict pretending to be the cloud | A second real Qdrant deployment holding `aegis_points` and `aegis_oplog`; the same code reaches a Qdrant Server by URL |
| Random scores when the GPU tier is absent | Escalation declined, with the reason recorded |
| Synthesised sensor readings | `psutil`, and **"no thermal or battery sensors on this platform"** when the platform has none |
| Seeded demo memories at boot | The node ships **empty**; the demo and tests ingest their own corpus through the same path a device uses |
| A frontend that invents numbers when the backend is down | `NO NODE`, every figure blanked — a stale number is indistinguishable from a live one |

One clarification, since [the engineering log](ENGINEERING.md#31-deterministic-simulation--a-seed-for-every-bug)
describes a simulator. The deterministic
simulator does not stand in for anything: it runs **the real
`GossipAgent`, the real CRDT, the real wire codec and the real signature
verification**, and replaces exactly two things — the wall clock and the
random number generator. That is the whole point. A mock mesh would find bugs
in the mock. The bugs in rows 13–15 of [the defect table](ENGINEERING.md) are bugs in shipped code, each with a
regression test that fails against the code as it was.

Two things this environment could not run, stated rather than papered over:

- **A remote Qdrant Server.** The adapter, wire format and configuration are
  real and exercised against embedded Qdrant; the org egress policy blocks the
  hosts the server binary and image come from, so the *remote* round trip is
  untested here.
- **Triton.** The client is real gRPC ensemble code; it needs a reachable
  server, which this environment has no GPU for.

## 11. Frontend

`frontend/index.html` — zero dependencies, zero build step. Open it, or serve
the folder statically.

- A **Windows-XP-era boot sequence** cold-starts the node: POST text, a
  chunked-block progress bar, then the haze resolves into the console.
- Black and orange, heavy grain, scanlines, drifting haze, deliberately
  off-grid typographic positioning.
- Live **node console**: link state, memory tiers, sync cursor, execution
  provider, event stream, and a working hybrid-search box.
- Talks to the backend at `window.AEGIS_API` (defaults to
  `http://localhost:8000`) over REST + WebSocket. With no backend reachable it
  flips to `LOCAL / SIMULATED` and stays fully explorable — which is, after
  all, the entire point of an offline-first product.

```bash
cd frontend && python3 -m http.server 5173   # → http://localhost:5173
```

Point it at a live backend:

```html
<script>window.AEGIS_API = "http://192.168.1.42:8000";</script>
```

---

# Appendix 3 — how every part actually works

The main architecture is §5 and the feature plan is §6. This appendix is the
layer beneath both: every module in the node, what mechanism it uses, and why
that mechanism rather than the obvious one. It is written to be read by
somebody deciding whether to trust the system, so where a choice was made
against the intuitive option, the reason is stated rather than implied.

A hundred and three modules (excluding `__init__.py`), grouped as they are on
disk.

## A3.1 Core — clocks, scheduling, budgets, isolation

**`core/clock.py` — hybrid logical clock.** Wall time alone cannot order
events across devices whose clocks disagree, and a Lamport counter alone
cannot be compared to a human timestamp. An HLC carries both: a physical
millisecond and a logical counter that increments when two events share a
millisecond. It never goes backwards, even when NTP steps the system clock
backwards underneath it, because the physical component is `max(local wall,
last seen)`. Every memory and every operation carries one, and that is what
makes "which of these two edits is later" answerable offline.

**`core/bus.py` — in-process pub/sub, bounded and drop-oldest.** Every
subsystem publishes what it did; the WebSocket gateway and the console
subscribe. Queues are bounded and drop the *oldest* event when full, which is
the correct choice for telemetry: a slow consumer must never apply
backpressure to the thing it is watching, and a stale event is worth less than
a current one.

**`core/scheduler.py` — QoS lanes.** Background work and queries compete for
one interpreter, so background work is submitted to lanes (`QUERY`, `SYNC`,
`MAINTENANCE`) with deadlines and a shedding policy rather than run
unconditionally. Under pressure the maintenance lane is dropped first, which
is why an index rebuild or a scrub cannot make a query slow. Pressure is
exposed so the SLO ladder can read it.

**`core/slo.py` — the degradation ladder.** Five rungs, FULL through SURVIVAL.
Each optional retrieval stage declares the rung at which it stops running, and
the pipeline asks once per query so the answer cannot shift mid-flight. The
manager climbs on error-budget burn rate or scheduler pressure, one rung per
evaluation, and holds a rung for a dwell period so a burst cannot make it
flap. It de-escalates only when burn falls and pressure clears. This is the
subsystem that was found blind — see [the engineering log](ENGINEERING.md) —
because latency was reported to it
only from the HTTP layer.

**`core/supervisor.py` — restart budgets.** Background loops are supervised
with a restart budget and crash-loop detection. A subsystem that dies
repeatedly is reported, not restarted forever, because an endless restart is
indistinguishable from working until somebody looks.

**`core/tenancy.py` — isolation that is enforced structurally.** Tenants have
quotas (points, bytes, ingest rate, QPS), hashed and scoped API keys, and a
visible id set. Cross-tenant reads are refused in `MemoryStore.get`, not at
the call sites: one forgotten filter upstream is a data breach, one refusal at
the bottom is not.

**`core/energy.py` — joules per operation.** Three sources tried in order —
the CPU package's RAPL counter, battery discharge, then CPU-seconds times an
assumed per-core draw — and every reading carries which one produced it. The
first two are measurements and the third is a model, labelled `modelled`
wherever it appears. Attribution uses CPU time rather than wall time, so a
query that waited on the micro-batcher is not billed for the wait.

**`core/provenance.py` — the receipt.** A Merkle tree (not a flat hash, so a
mismatch can be *located*) over the source files, the model weights and the
compiled graphs. `verify_against` names changed, added and removed files. The
receipt states its own limit: integrity, not authenticity.

**`core/metrics.py`, `core/tracing.py`.** Counters, gauges and
streaming-quantile histograms with Prometheus exposition; spans nested per
query so `/api/v1/traces` can show where a slow query actually spent its time
rather than where one would guess.

**`core/circuit.py`, `core/backoff.py`, `core/ratelimit.py`.** A per-endpoint
circuit breaker so a dead coordinator is not retried into the ground;
decorrelated-jitter backoff, which avoids the synchronised retry storm that
plain exponential backoff produces across a fleet; token buckets for sync
bandwidth and for fleet-wide reconnect control.

**`core/ids.py`.** ULID-shaped identifiers: monotonic and lexicographically
sortable, so an id sorts by creation time without a separate index.

**`core/errors.py`.** Typed failures. A subsystem that fails loudly is cheaper
than one that lies — `LinkUnavailable`, `WalCorrupt`, `PolicyDenied`,
`StorageLocked` each carry the remedy in the message.

## A3.2 Memory — the system of record

**`memory/schema.py` — what a memory is.** `MemoryPoint` carries text, dense
and sparse vectors, payload, tier, sensitivity, **sync class**, HLC, device
id, tenant, model version and supersession. `SyncClass` is ordered by
restriction (`FULL` < `REDACTED` < `METADATA_ONLY` < `LOCAL_ONLY`) with a
`strictest()` helper, because a memory crossing devices must never have its
handling relaxed by the receiver.

**`memory/wal.py` — crash-safe write-ahead log.** Every mutation is appended
with a CRC and fsynced *before* the write is acknowledged. Replay stops at the
first torn record and reports how many it discarded. A torn tail is not data
loss: the log is fsynced before acknowledgement, so a half-written record is
one whose caller never received an answer, and replaying it would invent a
memory nobody was promised.

**`memory/segments.py` — immutable segments, crash-safe manifest.** Sealed
segments are content-addressed and never mutated. The manifest is written
temp → fsync → rename → fsync-directory, which is the only sequence that is
atomic on POSIX. `fsck` walks every segment and verifies checksums; `scrub`
does it in the background on a budget; a corrupt segment is quarantined rather
than deleted. `IncompatibleFormat` is a distinct exception because
`json.JSONDecodeError` subclasses `ValueError` and was once caught by the
version check, turning a corrupt manifest into a crash.

**`memory/store.py` — the ingest path.** Quota check, classify, redact,
embed, WAL append, index, extract graph facts, record the CRDT op. Quota is
checked first so an over-quota tenant does not waste an embed and leave
partial state. Nothing becomes searchable before it is in the WAL.

**`memory/vectorstore.py` — the store abstraction.** `NativeStore` (tiered,
adaptive, on-disk cold tier) and `QdrantStore` (embedded by default, a server
by URL). Embedded Qdrant is single-writer, so a second node on one data
directory raises `StorageLocked` naming the remedy rather than a bare
`BlockingIOError` from the lock library.

**`memory/index.py` — one collection's whole retrieval surface.** Dense index,
sparse postings, payload index, planner, tier map, warm int8 codes and cold
RaBitQ codes. The cold path scans codes in RAM and pages in only the shortlist
the error bound cannot rule out.

**`memory/ann.py` — the adaptive index and its cost model.** Flat BLAS, HNSW
or IVF-PQ, chosen by a model calibrated on the device at boot. The crossover
is solved from two measurements — cost per scanned point, cost per graph hop
*including the interpreter bookkeeping a real traversal pays* — with a margin,
because exhaustive search is also exact and a graph only earns the switch when
it is decisively faster. Migration is decided on the write path and performed
on the maintenance lane, because building a graph inline once stalled a single
write for 30.8 seconds.

**`memory/hnsw.py`.** Heuristic neighbour selection rather than plain nearest
neighbours, which keeps the graph navigable instead of clumping; soft deletes
with graph repair, so a removal does not strand the nodes that routed through
it.

**`memory/pq.py`.** OPQ rotation then IVF-PQ. `nprobe` and rescore depth are
self-calibrated against the actual corpus, because a true neighbour in an
unprobed cell cannot be recovered at any rescore depth — clustered corpora
need roughly a quarter of cells probed, uniform ones nearly all, and the right
number is a property of the data rather than the algorithm.

**`memory/quantize.py`, `memory/rabitq.py`.** Symmetric int8 for the warm
tier. The cold tier is RaBitQ: centre on the corpus centroid, rotate (fast
Hadamard when the dimension is a power of two, dense QR otherwise), keep sign
bits plus one float for how well the code aligns with its vector. That float
makes the estimator unbiased and yields a per-vector error bound, which then
*derives* the rescoring depth instead of a fixed multiplier. `ColdCodebook`
stores bits, factors and radii in three growable buffers so a cold scan runs
off views rather than allocating the corpus per query.

**`memory/growable.py`.** A row-appendable matrix with geometric capacity that
exposes its filled region as a **view**. `np.vstack` per insert is O(n) per
write and O(n²) overall; this is amortised O(1) and keeps every `matrix @
query` one BLAS call over contiguous memory. Deletion is swap-with-last, and
the caller is handed the moved row so it can fix its own bookkeeping.

**`memory/vectors.py`, `memory/tiering.py`.** HOT full precision resident,
WARM int8, COLD evicted to a memory-mapped file with only codes in RAM.
Keeping a full-precision copy "for rescoring" would make the compression ratio
a slide rather than a fact. The compactor promotes and demotes on a decayed
access score.

**`memory/filters.py`, `memory/planner.py`.** Inverted payload indexes over
the fields worth indexing, and a cost-based planner that chooses pre-filter,
post-filter or scan from measured selectivity — the same decision a database
optimiser makes, for the same reason.

**`memory/graph.py` — the bitemporal knowledge graph.** Two independent time
axes: when a fact was *true* (valid time) and when this device *believed* it
(transaction time). Retraction is not deletion, so "when did we stop believing
this" remains answerable, which is exactly what an incident review asks. The
live view is materialised once and reused until a mutation bumps a version or
a fact's validity window closes — the earliest such moment is kept as an
expiry so the cache is correct rather than merely fast. Time-travel queries
bypass it deliberately: a cache of *now* must never answer a question about
some other time.

**`memory/consolidation.py`, `memory/repair.py`.** Near-duplicate memories are
merged with provenance preserved. Repair recovers damaged points peer-first
and reports what neither peer could supply as permanently lost, by identifier
— a corrupt segment is usually a lost durable copy rather than a lost memory,
and reporting those as "recovered from a peer" would overstate the repair.

## A3.3 Inference — real weights, and a node that checks its own representation

**`inference/models.py` — provisioning without a network.** The pretrained
32000×256 token table and its 32k BPE tokenizer ship inside the PyPI wheel. At
first boot the node compiles two ONNX graphs from those weights, writes the
tokenizer and a provenance file, and content-addresses both graphs into the
registry. Two details that cost an evening each: `onnx` emits IR version 14
while the runtime accepts at most 13, so the IR version is pinned; and
`ReduceL2` takes its axes as an *attribute* until opset 18 while `ReduceSum`
takes them as an *input* from opset 13, which is not a symmetry anyone
expects.

**`inference/onnx_runtime.py` — sessions and the execution-provider ladder.**
Providers are probed in preference order and the best available is used; the
optimised graph is serialised on first boot so later boots pay nothing. The
embedding graph is a gather, a masked mean and an L2 normalise — which is what
makes weighted pooling possible with no graph change, since the mask is the
denominator. Text is clipped to a character bound before tokenising, because
only `max_tokens` survive and a 1 MB input otherwise spends 694 ms producing
tokens it immediately discards.

**`inference/batcher.py` — micro-batching, on the event loop deliberately.**
Concurrent embeds inside an 8 ms window coalesce into one run. Moving that run
to a thread pool was tried and measured *worse* (2,069 → 1,332 qps at 256
concurrent); ONNX Runtime already releases the GIL and parallelises
internally, so the executor added a hop per batch and bought nothing. The
measurement is in the module docstring so it is not rediscovered.

**`inference/embedder.py`.** The dense encoder, plus the node's two on-device
adaptations and the gate that decides whether either is allowed on.

**`inference/lexicon.py` — unigram statistics from this device's own stream.**
SIF weighting needs `p(t)`, and estimating it here rather than shipping it
from a web crawl is what makes it an edge feature. It also measured *worse* on
this encoder at every setting, because the token table is a distillation
trained for mean pooling and has already absorbed the correction. It ships
off, kept as the candidate the gate rejects.

**`inference/geometry.py` — corpus-fitted whitening.** Streaming mean and Gram
matrix, Ledoit–Wolf shrinkage so the node never inverts an estimate it cannot
support, eigendecomposition, and a **rank** rather than a dimension: enough
directions to hold the energy target, and never one whose eigenvalue has
fallen through a conditioning floor. Whitening at full dimension amplifies
near-zero directions by four orders of magnitude and took cold-tier recall
from 0.77 to 0.11.

**`inference/adaptation.py` — the gate.** A paraphrase task built from the
node's own memories: delete half a stored document's words and the document it
came from is the correct answer by construction, so no labels are needed.
Adoption requires a **paired bootstrap** whose 95% interval excludes zero,
because the same transform measured +0.0175 MRR on one probe sample of one
corpus and −0.0001 on another. Measuring never arms anything: the winning
transform changes the output dimension, so arming is an explicit call behind a
renewal migration.

**`inference/sparse.py`, `inference/reranker.py`.** A SPLADE-shaped sparse
encoder backed by BM25 statistics, and a ColBERT-style late-interaction
reranker computing MaxSim over per-token vectors from the same pretrained
space, returning the token alignments that produced the score so a ranking can
be explained rather than asserted.

**`inference/classifier.py` — PII on the ingest path.** Anchored patterns and
an overlapping-window scan. It sits on the ingest path by design, which is
also why its worst case was a weapon: one unanchored regex made a single 1 MB
write take 79 minutes at 100% CPU. Truncating the scan was rejected — a secret
at offset two megabytes must not escape classification because scanning it was
inconvenient.

**`inference/governor.py`, `inference/registry.py`, `inference/triton.py`.**
A thermal and power governor that sheds inference cost on its own terms rather
than waiting for the silicon to throttle it, and reports **unavailable**
where a platform exposes no sensor instead of synthesising one; a
content-addressed model registry; and a real gRPC Triton client whose
escalation decision is recorded with its reason, including when it declines.

## A3.4 Retrieval — the query path, stage by stage

**`retrieval/pipeline.py`** runs: understand → *exact cache* → embed →
plan → dense + sparse → fuse → rerank → graph boost → diversity → conformal →
score. Each optional stage asks the SLO ladder once whether it may run.

**`retrieval/query_understanding.py`.** A BK-tree over the local vocabulary
repairs spelling with no network and no spell-check service; corpus
co-occurrence expands the query; intent is classified and may *suggest* a
collection. The BK-tree prunes on the triangle inequality, which requires true
edit distances — pruning on clamped distances silently returns nothing, as it
did until a test caught it. An inferred narrowing is advisory: when it matches
nothing the query is retried at the caller's scope, because a guess that
silently replaces results with an empty page is worse than no guess.

**`retrieval/cache.py` — two layers.** An exact layer keyed on normalised
query text, checked *before* the encoder, and a vector layer for
differently-worded questions. A vector hit teaches the exact layer, so the
next identical question skips the encoder — a cache that only learns from full
work never learns from itself. Both share one epoch, so a write invalidates
both. The key carries tenant, collection, mode, k and a canonical digest of
the filter, because everything that changes what a correct answer *is* belongs
in the key.

**`retrieval/fusion.py`, `retrieval/scoring.py`.** Reciprocal rank fusion,
which combines rankings without needing the two score scales to be
commensurate; then final ranking over rerank score, recency with a half-life,
confidence and the adapter delta.

**`retrieval/diversity.py`.** Maximal marginal relevance, so five phrasings of
one memory do not occupy all five result slots.

**`retrieval/conformal.py`.** Split conformal prediction: a held-out
calibration set gives a distribution-free coverage guarantee, and the node
abstains when the answer falls below it. Realised coverage is reported
alongside the target, because a guarantee nobody checks is a claim.

**`retrieval/contradiction.py`, `retrieval/agent.py`.** Contradiction
detection between retrieved memories, and an agentic loop that plans,
retrieves, verifies and answers with citations — every claim pointing at the
memory it came from, and the reasoning steps returned with it.

## A3.5 Sync — designed for a link that is usually down

**`sync/crdt.py`.** An append-only operation log of commutative operations.
Commutativity is what makes order-independent merge possible, which is what
makes offline editing safe.

**`sync/merkle.py`, `sync/iblt.py`.** Merkle range digests to find *which*
ranges differ without sending the contents, then an Invertible Bloom Lookup
Table to recover the exact symmetric difference in one round — cells sized to
the expected divergence, so reconciliation costs a function of what actually
differs rather than of how much is stored.

**`sync/causal.py`.** Vector clocks and a causal delivery buffer, applied to
the rumour path only. CRDT operations are commutative so anti-entropy can
apply a set directly; the streaming path needs ordering because a supersede
can outrun the thing it supersedes. Clocks tick only for operations that are
actually shareable — a withheld operation that advanced the clock once stalled
delivery for everything behind it.

**`sync/gossip.py`, `sync/meshlink.py`.** Epidemic anti-entropy between peers,
preferring live peers while occasionally probing a dead one so partitions
heal. Both inbound paths — push and the pull half of anti-entropy — run through
one `_admit` check, which was not always so: the same restricted memory used to
be refused when offered and accepted when asked for. Admission is policy *and*
signature, enforced by the receiver on its own behalf. Egress filtering on the
sender is correct for an honest peer and worth nothing against a compromised
one, and a mesh of field devices is a population where a lost handset is a
Tuesday. `MeshLink` dispatches in-process for tests and simulated fleets;
`HttpMeshLink` carries the same RPC to another node process, which is what
makes two real devices possible. The peer receives it on
`POST /api/v1/mesh/exchange` and hands it to its own `GossipAgent`, so there
is one implementation of anti-entropy rather than two to keep in step.

**`core/invariants.py`.** The simulator's seven invariants, checked against
the live node on a two-second loop. Each advances a rolling cursor through its
own subject and spends a fixed budget, so the cost is bounded by the budget
rather than by how much the device has remembered — asserted by a test that
fails if a 4,000-operation log costs materially more per tick than a 64-operation
one, and another that fails if the cursor never moves. Local checks are weaker
than simulated ones, because a node sees only itself, and each carries the
scope of what it can actually establish rather than implying a guarantee about
the mesh. A violation is counted and published, never raised: a node that
halted on a detected inconsistency would turn a partial fault into a total one.
A check that throws is itself recorded, because a climbing assertion counter
with nothing behind it looks exactly like health.

**`sync/identity.py`.** An Ed25519 key pair per device, persisted beside its
data, signing the immutable part of every operation it creates: `op_id`,
`kind`, `point_id`, `hlc`, `device_id`, `body`, over canonical sorted-key JSON.
`ts` sits outside the signature on purpose — a relay may touch routing, not
content — and a test fails if that set is widened without meaning to. Keys are
learned trust-on-first-use — or, where a fleet has been enrolled, only on a
certificate binding an id to a key and signed by the fleet root, which closes
the first-contact gap TOFU cannot (`scripts/enrol.py`, and
[Enrolment](ENGINEERING.md#33-enrolment--closing-the-gap-the-signing-work-named)). Either way a
later change to a known identity's key is refused as an `identity_conflict`; keys travel alongside operations so a node
can verify an author it has never met, which is what stops signing from
breaking the partition tolerance it exists to protect. 62 µs to sign, 129 µs to
verify, 88 bytes on the wire. The limit it does *not* close — an attacker
present at first contact — is in the module docstring, not just here.

**`sync/compression.py`.** Delta encoding against earlier operations *in the
same frame*, int8 vectors, then zlib — skipped when it would not pay, and the
skip counted. The frame-local part is load-bearing and was not always true: the
elision context used to live on the encoder and outlive the frame while the
decoder started empty on each one, so every repeat of a low-cardinality field
was sent as a hole and dropped on arrival. One of those fields is
`sensitivity`, which is what a receiving node reads to decide whether it may
hold the memory at all. Cross-frame state cannot work over a transport that
drops, reorders and duplicates frames; a frame now carries its own context, and
a hole with nothing to fill it from raises rather than passing quietly.

**`sync/queue.py`, `sync/oracle.py`.** A durable queue that survives restart,
and a connectivity oracle with asymmetric EWMA — quick to believe the link is
down, slow to believe it is back — which fires `on_link_restored` the instant
it recovers rather than waiting for the next poll.

**`sync/transport.py`, `sync/conflict.py`, `sync/engine.py`.** Loopback, HTTP
and Qdrant transports behind one protocol; an arbiter that resolves by HLC,
then by vector similarity, and escalates a genuine semantic conflict to a
human review queue rather than silently picking; and the engine that coalesces
concurrent reconciliations, pushes what policy allows, pulls what it is
missing and materialises it — preserving the sender's handling class rather
than relabelling everything it accepts as freely shareable.

## A3.6 Learning, renewal, policy, chaos

**`learning/adapter.py`.** A rank-16 low-rank adapter over the retrieval
space, trained from real choices rather than synthetic labels. Low rank
because the update must be small enough to send and cheap enough to apply on a
query.

**`learning/privacy.py`, `learning/federated.py`.** Per-coordinate Gaussian
noise with a tracked budget, and secure aggregation with pairwise cancelling
masks so the coordinator sees only the sum. Each round reports its SNR and the
cohort size the stated epsilon would actually require — which, at ε=2 over 12k
parameters, is hundreds of thousands of devices. DP can be switched off for a
small fleet; that is an audited choice, and secure aggregation still hides the
individual update either way.

**`renewal/freshness.py`, `renewal/migrator.py`, `renewal/scheduler.py`.**
Freshness scoring over age, access and drift; dual-space migration that
re-embeds into a new space while both are queryable, with shadow evaluation
comparing the two before the switch and checkpoints so an interrupted
migration resumes rather than restarts.

**`policy/engine.py`, `policy/redaction.py`, `policy/audit.py`.** A declarative,
hot-reloadable policy engine that resolves its file against the working
directory and then the package root, and marks the snapshot
`using_built_in_default` when it found neither — a policy that silently fell
back to defaults because the working directory differed is a policy that is
not enforcing what anyone thinks. Redaction is reversible through a local-only
vault, so the original never leaves but is not destroyed. The audit log is
hash-chained, and its verification walks the chain rather than trusting a
stored flag.

**`chaos/faults.py`.** Twelve faults — link drop, packet loss, latency spike,
clock skew, disk full, thermal spike, corrupt WAL, partition, corrupt segment,
memory pressure, query storm, peer churn — each with a scheduled clear, so the
resilience claims can be tested rather than asserted. `POST /chaos/kill` is
separate and refuses unless a supervisor is present.

## A3.7 API and the composition root

**`node.py`** builds every subsystem and supervises the background loops:
compaction (which also runs deferred index migrations and the periodic space
review), consolidation, governor sampling, telemetry, archiving, scrubbing,
SLO evaluation, sync and mesh. `close()` releases external handles and marks
the transports unavailable, so a reconcile still in flight sees a link that is
down — something every path already survives — rather than a bare
`RuntimeError` raised into a task nobody is awaiting.

**`api/`** is nineteen routers over that node, plus a multiplexed WebSocket
gateway. `security.py` resolves a principal from a scoped, hashed API key and
refuses cross-tenant access structurally; auth is off by default because a
single-tenant device on a private network should not need a credential to
answer its own operator, and every route behaves identically with it on.


---
*Code Cubicle 6.0 · 3 OCT online · 11 OCT offline*

## A3.8 Simulation — the fleet as a pure function

**`core/determinism.py`.** The ambient environment, swappable. `Environment`
is the real one: `time.time()`, `time.perf_counter()`, an unseeded generator.
`VirtualEnvironment` is the simulated one, and the only way its clock moves is
`advance()` or `skew()` — the latter allowed to go *backwards*, because devices
do that and a simulator that cannot express it cannot find the bug. `sleep()`
advances the clock and returns immediately, so a simulated hour costs nothing.
`simulated(seed)` is a context manager that installs the virtual environment
and restores the real one on exit, so a test that simulates cannot leak a
frozen clock into the test after it.

Everything downstream calls `determinism.now()`, `determinism.monotonic()`,
`determinism.rng()` and `determinism.sleep()` rather than the stdlib. That
indirection is the entire mechanism: it is what makes an execution replayable,
and it costs one attribute lookup.

**`core/ids.py`.** ULID-shaped identifiers, drawing their timestamp and
randomness from the environment rather than from `time` and `os.urandom`. This
is not stylistic: an operation id is what the IBLT hashes and what a fetch is
sorted by, so ids outside the seeded execution make *reconciliation*
non-deterministic while everything else replays perfectly — which presents as
a sweep returning 454 failures and then 456. The per-millisecond counter is
module state, so `simulated()` resets it on the way in and on the way out.

**`scripts/audit_determinism.py`.** An AST pass over every simulated module,
failing the build on a direct call to the clock or the generator, on
`os.urandom`, and on a module that reaches for `determinism` without importing
it. Exemptions are named with their reason — a codec has no clock, `meshlink` measures real RTT to
a real peer, network clients own their own timeouts, and `determinism.py`
*is* the environment. A single missed call does not fail a test; it makes a
seed stop reproducing, silently, which is worse.

**`sim/world.py`.** The world: N peers, each with a real `GossipAgent`, a real
CRDT store and a real signing identity whose key material is derived from the
seed so that keys replay too. Steps are drawn from a weighted population of ten
actions — write, gossip, partition, heal, skew, crash, duplicate, idle, and
three Byzantine ones — and seven invariants are checked after every single
step. `replay_step` performs a recorded action *without consulting the
generator*, which is what makes shrinking possible: re-rolling the dice for a
replayed step would produce a different history and a different answer, leaving
the shrinker's question unanswerable.

Two details worth stating because they were both wrong first:

- The `crash` action originally decided what was lost by reading the
  simulation's shadow stores and ignoring each agent's own view. That is a bug
  in the *model*, and the simulator's first finding was its own.
- `check_converged` counts what honest devices wrote. Operations the world
  knows were fabricated by an attacker are excluded, because a forgery that
  fails to spread is the system working, and counting it read a successful
  defence as a missing replica.

**`scripts/simulate.py`.** The driver. Sweeps seeds, and on a failure runs
delta debugging — remove one action, replay, does it still fail? — down to the
shortest history that reproduces it. `--replay N` re-runs one seed and prints
its history. `--unsigned` runs the control. `--no-shrink` counts failures
without reducing each one, for a control run where the rate is the result. It
reports fleet-days simulated and prints, in its own output, that a clean sweep
is not a proof: it is *no counterexample in N executions*, with N stated.
