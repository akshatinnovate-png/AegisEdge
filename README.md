# AegisEdge

**AI-Powered Edge Memory & Intelligence Platform**
Code Cubicle 6.0 — **Problem Statement 03**

> An offline-first edge brain. It remembers locally, retrieves in single-digit
> milliseconds without a network, decides for itself what may leave the device,
> and heals its own state the instant connectivity returns.

---

## 0. The thesis

Most "edge AI" demos are a vector database running on a laptop. That is not an
edge system — it is a cloud system with the cable unplugged.

A real edge node has to survive things a cloud node never sees: the power dies
mid-write, the link flaps every ninety seconds, the embedding model is upgraded
while 400k vectors are already on disk, two devices edit the same memory while
partitioned, and the thermal governor halves your clock speed in the middle of a
query. AegisEdge is designed around those failures, not around the happy path.

The stack is built on **Qdrant Edge** for on-device semantic memory, **ONNX
Runtime** for local inference, and **NVIDIA Triton** for the heavy cloud tier,
joined by a sync protocol that treats disconnection as the normal state and
connectivity as the exception.

---

## 1. System shape

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

## 2. Backend feature plan

### 2.1 Memory core — Qdrant Edge

| Feature | Detail |
|---|---|
| Embedded collections | Qdrant Edge runs **in-process**, no sidecar, no network hop. Collections are sharded by *memory class* (`episodic`, `semantic`, `procedural`, `sensor`). |
| Hybrid vectors | Every point carries a **dense** vector (`bge-small-en-v1.5`, 384d) *and* a **sparse** vector (SPLADE-mini / BM25 fallback) so lexical rare-token matches survive offline. |
| Named vector spaces | Multi-vector points: `text`, `vision`, `audio`, `fused` — one point can be retrieved through any modality. |
| Scalar + binary quantization | INT8 scalar quantization on the hot tier, **binary quantization** on the cold tier for a 32× memory drop, with rescoring from the full vectors on disk. |
| Payload indexing | Keyed indexes on `ts`, `geo`, `device_id`, `sensitivity`, `model_version`, `ttl` to keep filtered search pre-filtered rather than post-filtered. |
| Memory tiering | **Hot** (mmap'd, full precision) → **Warm** (INT8) → **Cold** (binary + on-disk payload) → **Evicted** (sync'd to cloud, tombstone kept). A background compactor moves points across tiers on an access-recency × salience score. |
| Snapshotting | Periodic consistent snapshots to local object storage; a corrupted segment restores from the last snapshot + WAL replay instead of a full resync. |

### 2.2 Local inference — ONNX Runtime

| Feature | Detail |
|---|---|
| Everything is ONNX | Embedder, reranker, sensitivity classifier, intent router and the VAD/ASR front-end all ship as `.onnx` graphs. No Python model code at runtime. |
| Execution-provider ladder | Runtime probes and falls back: **TensorRT → CUDA → OpenVINO → CoreML → NNAPI → XNNPACK → CPU**. Chosen EP is reported in `/health` so the UI can show what silicon is actually being used. |
| Graph optimizations | `ORT_ENABLE_ALL`, ahead-of-time session serialization (`optimized_model_filepath`) so cold start is a load, not a re-optimization. |
| Quantized variants | Each model ships as `fp32 / fp16 / int8-dynamic / int8-static(QDQ)`. The **Thermal & Power Governor** hot-swaps variants at runtime when battery < 20% or package temp > 80°C — quality degrades gracefully instead of the node dying. |
| IOBinding + arena | Pre-allocated tensor arenas and zero-copy IOBinding to avoid per-request malloc; embeddings are produced into a reused pinned buffer. |
| Dynamic micro-batching | A 8 ms coalescing window batches concurrent embed calls into one session run — ~4× throughput on burst ingest. |
| Model registry | Content-addressed (`sha256`) local model store with signature verification before a model is ever loaded. Rollback is one pointer swap. |

### 2.3 Cloud inference — NVIDIA Triton

| Feature | Detail |
|---|---|
| Heavy tier | Large rerankers, VLM captioning, long-context summarization and the "deep reasoning" path run on **Triton** — reached only when the link is up and the policy engine allows it. |
| Ensemble models | Triton **ensemble** pipelines chain `tokenize → embed → rerank` server-side so one gRPC call replaces three round-trips over a bad link. |
| Dynamic batching | Triton dynamic batcher (`max_queue_delay_microseconds`) plus multiple model instances per GPU for fleet-wide throughput. |
| gRPC streaming | Bi-directional streaming inference so partial results reach the device as they are produced — a dropped link loses the tail, not the whole response. |
| Escalation policy | Local ONNX answers first and always. Triton is consulted only when local confidence < τ, the query is flagged complex, and RTT/jitter budget is met. Every escalation is logged with the reason, visible in the UI. |
| Model parity guard | Triton and ONNX embedders are version-locked. A mismatch triggers **renewal** (§2.6), never a silent mixed-embedding-space corruption. |

### 2.4 Approximate nearest neighbour — measured, not assumed

| Feature | Detail |
|---|---|
| HNSW | Full hierarchical graph with **heuristic neighbour selection** (naive top-M builds hubs and collapses recall on clustered data), bidirectional pruning, soft deletes with graph repair, entry-point demotion. **0.993 recall@10** measured. |
| OPQ + IVF-PQ | Product quantization on IVF residuals with a learned **OPQ rotation** — plain PQ slices by position, which assumes variance is already evenly spread; embeddings are nothing like that. **48-64x compression** at 0.993 recall. |
| Self-calibration | `nprobe` and rescore depth are *measured per corpus*, not guessed: clustered data needs ~24% of cells probed, uniform data ~90%. The index reports both. |
| Cost model | The node microbenchmarks its own silicon at boot and derives the flat→HNSW→IVF-PQ crossovers from it. A collection migrates strategy as it grows; migrations are logged. |
| Cold tier on disk | Cold vectors are evicted to a **memmap**; only 1-bit codes stay resident and rescoring pages in the shortlist alone. |

### 2.5 Query planning

| Feature | Detail |
|---|---|
| Payload index | Keyword postings plus sorted numeric arrays, carrying cardinality statistics so selectivity is *estimated* before anything executes. |
| Cost-based planner | Chooses pre-filter (resolve ids, scan that subset exactly), post-filter (ANN first, over-fetching by inverse selectivity) or full scan — and explains the choice in the response. |
| Provable emptiness | A filter that matches nothing does **zero** vector work. |
| Cross-collection | A `*` search aggregates per-collection plans instead of letting an empty one veto the query. |

### 2.6 Query understanding — local, in under 3 ms

| Feature | Detail |
|---|---|
| BK-tree spelling repair | Metric tree over the *corpus* vocabulary, so "colent presure" becomes "coolant pressure" — domain terms a generic dictionary would never hold. Common English words are protected from over-eager correction. |
| Co-occurrence expansion | PMI-style expansion learned from ingested text; no embedding round trip. |
| Unit & temporal normalisation | `4.2mm/s` is normalised; "from the last 2 hours" becomes a payload filter the planner can use. |
| Intent routing | Procedural / sensor / episodic / semantic routing from the query shape. |

### 2.7 Retrieval & reasoning

- **Hybrid fusion** — dense + sparse candidates merged with **Reciprocal Rank Fusion**, then cross-encoder reranked on-device.
- **Late interaction** — ColBERT-style **MaxSim** over per-token vectors (int8-quantized) on the shortlist only, with query-term → document-term alignments returned as evidence.
- **On-device adapter** — a rank-16 low-rank adapter (~48 KB) trained from real feedback by contrastive updates, so the node learns *this* site's vocabulary without a fine-tune.
- **Span tracing** — every query emits a nested span tree and names its own hotspot.
- **Temporal decay + salience** — final score = `α·similarity + β·recency_decay + γ·access_frequency + δ·pinned` so stale memories sink without being deleted.
- **Contradiction detection** — an NLI head flags memories that contradict newer ones; the loser is *superseded*, not erased, and the chain stays inspectable.
- **Memory consolidation** — a nightly (or idle-triggered) job clusters near-duplicate episodic points and distills them into a single semantic point, with provenance links to the originals. Local memory stops growing linearly with uptime.
- **Agentic loop** — plan → retrieve → (optionally escalate to Triton) → verify → answer, with every step emitted on the telemetry bus so the frontend can render the node's actual reasoning trace.
- **Query cache** — semantic cache keyed on embedding proximity; a near-identical question answers from cache in <1 ms.

### 2.8 Sync engine — edge ⇄ cloud

| Feature | Detail |
|---|---|
| CRDT log | Every mutation is an **LWW-Element-Set / OR-Set** operation with a hybrid logical clock (`HLC`), so two partitioned devices converge without a coordinator. |
| Delta sync | Devices exchange **Merkle-tree range digests** first; only divergent ranges transfer. A 400k-point collection reconciles in kilobytes. |
| Bidirectional | Edge→cloud pushes new local knowledge; cloud→edge pulls fleet knowledge relevant to *this* device's geo/role/task profile (not the whole corpus). |
| Selective sync | The **policy engine** (§2.7) decides per-point: `local_only`, `sync_metadata_only`, `sync_full`, `sync_after_redaction`. |
| Conflict arbiter | Resolution ladder: HLC → device trust score → semantic merge (both retained, linked as variants) → human review queue surfaced in the UI. |
| Backpressure | Sync respects a token-bucket bandwidth budget and yields to foreground queries — sync never makes the device feel slow. |
| Resumable transfer | Chunked, checksummed, offset-resumable. A link that dies at 93% resumes at 93%. |
| Tombstones + GC | Deletes propagate as tombstones with a grace window, then are garbage collected fleet-wide. |

### 2.9 Peer-to-peer mesh — no cloud involved

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

### 2.10 On-device learning

| Feature | Detail |
|---|---|
| Retrieval adapter | Rank-16 projections over the embedding space, trained by contrastive hinge from click feedback. ~48 KB, microseconds per example. |
| Differential privacy | Clipped updates plus calibrated Gaussian noise, with a tracked (ε, δ) budget the node refuses to overspend. |
| Secure aggregation | Pairwise masks cancel exactly in the sum, so the coordinator sees the average and never an individual update. Rounds with too many dropouts are abandoned rather than corrupted. |
| Honest utility reporting | Each round reports its SNR and the cohort size that epsilon would actually need. DP is switchable for small fleets — an audited choice, not a silent one. |

### 2.11 QoS scheduling

Background work is not optional, but a waiting person outranks a re-embedding
batch. Work is admitted into priority lanes (interactive / sync / maintenance /
renewal) with deadlines; stale background jobs are **shed** rather than run
late, and admission control rejects work the node cannot finish instead of
missing every deadline at once.

### 2.12 Data renewal

Memory rots. Renewal is a first-class subsystem, not a cron job.

- **Freshness scoring** — every point carries `ttl`, `confidence`, `last_verified_at`. A decay function marks points `stale` before they mislead anyone.
- **Re-embedding on model upgrade** — when the embedder version bumps, a **dual-space migration** begins: both spaces are queried and results fused while a background job re-embeds the corpus in priority order (hot tier first). Zero read downtime, zero mixed-space corruption.
- **Progressive renewal** — re-embedding is checkpointed and interruptible; power loss resumes from the last checkpoint.
- **Source revalidation** — points linked to an external source are re-fetched and diffed when connectivity allows; changed sources supersede the old memory and keep the audit chain.
- **Compaction & decay** — expired points drop to cold, then to tombstone. Pinned and high-salience points are exempt.
- **Shadow evaluation** — before a renewed model is promoted, a golden query set is replayed against both spaces; promotion is blocked on recall regression.

### 2.13 Bitemporal knowledge graph

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

### 2.14 Calibrated confidence and abstention

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

### 2.15 Durability engineering

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

### 2.16 Multi-tenancy

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

### 2.17 Survival: the degradation ladder

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

### 2.18 Policy & privacy engine

- On-device **PII / sensitivity classifier** (ONNX) tags every chunk `public | internal | sensitive | restricted` before it is ever written.
- Declarative policy (`YAML`, hot-reloadable): *restricted never leaves the device; sensitive syncs only redacted; public syncs freely.*
- **Redaction pipeline** — named-entity masking with a reversible local-only vault so the device can still resolve what the cloud can never see.
- Encryption at rest (AES-256-GCM, key in TPM/Secure Enclave/keyring), mTLS in flight, per-device identity certificates.
- **Tamper-evident audit log** — hash-chained, append-only record of every read, sync and escalation.

### 2.19 Instantaneous reconnection

The part most projects hand-wave. Reconnection is sub-second and stateful.

- **Connectivity oracle** — active probes + OS network-change hooks + RTT/jitter/loss EWMA classify the link as `OFFLINE / DEGRADED / METERED / HEALTHY`. State changes fire in milliseconds, not on the next poll tick.
- **Pre-warmed transport** — QUIC/HTTP3 with **0-RTT session resumption** and a warm connection pool; reconnect skips the full handshake.
- **Session continuation tokens** — the server remembers the device's sync cursor, so resumption is `"continue from op 84,213"`, not a re-handshake.
- **Operation queue** — every action taken offline is durably queued, idempotency-keyed, and replayed in causal order on reconnect. Nothing is lost, nothing is applied twice.
- **Hedged requests** — during `DEGRADED`, duplicate requests are raced across paths and the first response wins.
- **Reconnect storm control** — decorrelated jitter backoff + fleet-wide token bucket so 10,000 devices returning at once don't DDoS the coordinator.
- **Circuit breaker** — per-endpoint breakers trip fast and half-open probe, so a sick cloud endpoint never stalls the local path.
- **Optimistic UI contract** — the WebSocket pushes `link_state` transitions so the frontend flips between LOCAL and FUSED modes the moment the link moves.

### 2.20 Realtime & transport layer

- **WebSocket multiplex** — one socket, logical channels (`telemetry`, `sync`, `search`, `reasoning_trace`, `alerts`), heartbeats with server-side liveness detection.
- **Event bus** — internal pub/sub; every subsystem emits structured events which the gateway fans out to subscribed UIs.
- **Server-Sent Events fallback** for locked-down networks; **gRPC** for device↔cloud; **REST** for control plane.
- **Backpressure-aware streaming** — slow consumers get sampled, not buffered to death.

### 2.21 Reliability & operations

- **Supervisor** with per-subsystem health, restart budgets and crash-loop detection.
- **WAL + crash recovery** — an unclean shutdown replays the write-ahead log; a half-written batch is never half-visible.
- **Chaos hooks** — inject link loss, packet loss, clock skew, disk-full and process kill from an admin endpoint. The demo can *prove* resilience live, on stage.
- **Observability** — OpenTelemetry traces spanning `edge → link → Triton → back`, Prometheus metrics, structured logs; p50/p95/p99 exposed to the UI.
- **Benchmark harness** — reproducible recall@k, latency and sync-convergence numbers, checked in.

---

## 3. Planned stack

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
| Frontend | Vanilla HTML/CSS/JS, zero build step (§5) |

---

## 4. API surface (frontend contract)

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
| `GET` | `/api/v1/mesh/status` · `POST /mesh/round` | Peer mesh membership and anti-entropy |
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
| `WS` | `/api/v1/stream` | Multiplexed live telemetry, sync events, reasoning traces |

Full surface at `/docs` once the node is running.

---

## 5. Running it

```bash
# backend — the node
cd backend
pip install -r requirements.txt
uvicorn aegis.main:app --port 8000      # REST + WebSocket on :8000
python3 scripts/demo.py                 # whole lifecycle in one process, no server
python3 scripts/bench.py                # index recall + latency, measured here
python3 -m pytest tests -q              # 211 tests

# frontend — the console
cd frontend && python3 -m http.server 5173
```

The console auto-detects the node on `http://localhost:8000`. `backend/README.md`
has the module map and the optional-dependency matrix.

What the demo script actually exercises, end to end: cold boot and WAL replay,
hybrid retrieval with the link down, policy blocking a restricted memory from
egress, twelve observations queued through an outage, reconnection replaying
them, fleet knowledge pulled back down, a contradiction detected and
superseded, a dual-space re-embedding migration with shadow evaluation, four
injected faults, local query repair and expansion, three planner decisions,
**a peer-to-peer mesh round with the uplink down**, and an adapter trained
from feedback with a privacy-budgeted federated contribution.

## 6. Frontend

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

## 7. Status

- [x] Problem statement locked — PS03
- [x] Backend architecture & feature plan
- [x] Frontend shell, live against the node
- [x] Memory core — tiered store, quantization, WAL, compactor, consolidation
- [x] ONNX embed / rerank pipeline — EP ladder, micro-batching, thermal governor
- [x] Sync engine — CRDT + Merkle deltas, durable queue, conflict arbiter, resumption
- [x] Policy, redaction vault, hash-chained audit
- [x] Renewal orchestrator — freshness, dual-space migration, shadow eval
- [x] Chaos harness · 132 tests
- [x] Real ANN — HNSW, OPQ/IVF-PQ, device-calibrated strategy selection
- [x] Cost-based query planner with payload statistics
- [x] Query understanding — BK-tree repair, expansion, intent, unit/time extraction
- [x] Late interaction (MaxSim) reranking
- [x] Peer-to-peer mesh — IBLT reconciliation, causal delivery, gossip
- [x] On-device learning — adapter, differential privacy, secure aggregation
- [x] QoS scheduler and span tracing
- [x] Benchmark harness (`scripts/bench.py`) — numbers in this README come from it
- [x] Bitemporal knowledge graph — extraction, time travel, multi-hop, graph-boosted retrieval
- [x] Conformal prediction — calibrated coverage, abstention, drift detection
- [x] Durability engine — content-addressed segments, crash-safe manifest, fsck, scrub, PITR
- [x] Self-healing repair — peer-first recovery, honest loss reporting
- [x] Multi-tenancy — structural isolation, namespaced cache, scoped keys, quotas
- [x] SLO degradation ladder — sheds stages to protect p99 under load
- [x] Survival suite — invariants asserted under simultaneous fault storms · 211 tests
- [x] Triton escalation tier (client + policy; needs a live endpoint to light up)
- [ ] Qdrant Edge wheel pinned in CI (adapter is in, falls back to the native store)
- [ ] Multi-modal named vector spaces (schema supports them; encoders pending)

---

*Code Cubicle 6.0 · 3 OCT online · 11 OCT offline*
