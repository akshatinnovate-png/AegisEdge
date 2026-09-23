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
┌──────────────────────────── EDGE NODE (offline-capable) ────────────────────────────┐
│                                                                                     │
│  Ingest ─▶ Normalizer ─▶ Sensitivity Classifier ─▶ Chunker ─▶ ONNX Embedder         │
│                                    │                              │                 │
│                                    ▼                              ▼                 │
│                            Policy Engine (local/cloud)     Qdrant Edge (embedded)   │
│                                    │                       dense + sparse + payload │
│                                    ▼                              │                 │
│                          Write-Ahead Log (crash-safe)             ▼                 │
│                                    │                    Hybrid Retrieval Engine     │
│                                    ▼                     RRF ─▶ ONNX Reranker       │
│                            Sync Engine (CRDT deltas)              │                 │
│                                    │                              ▼                 │
│                                    │                      Agentic Reasoner          │
│                                    │                              │                 │
│  Supervisor · Telemetry Bus · WebSocket Gateway ◀──────────────────┘                │
└────────────────────────────────────┼────────────────────────────────────────────────┘
                                     │  intermittent, hostile, lossy link
                                     ▼
┌──────────────────────────── CLOUD TIER ─────────────────────────────────────────────┐
│  Sync Coordinator ─ Qdrant Server (canonical) ─ Triton Inference Server              │
│  Renewal Orchestrator ─ Fleet Registry ─ Conflict Arbiter ─ Object Store             │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

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

### 2.4 Retrieval & reasoning

- **Hybrid fusion** — dense + sparse candidates merged with **Reciprocal Rank Fusion**, then cross-encoder reranked on-device.
- **Temporal decay + salience** — final score = `α·similarity + β·recency_decay + γ·access_frequency + δ·pinned` so stale memories sink without being deleted.
- **Contradiction detection** — an NLI head flags memories that contradict newer ones; the loser is *superseded*, not erased, and the chain stays inspectable.
- **Memory consolidation** — a nightly (or idle-triggered) job clusters near-duplicate episodic points and distills them into a single semantic point, with provenance links to the originals. Local memory stops growing linearly with uptime.
- **Agentic loop** — plan → retrieve → (optionally escalate to Triton) → verify → answer, with every step emitted on the telemetry bus so the frontend can render the node's actual reasoning trace.
- **Query cache** — semantic cache keyed on embedding proximity; a near-identical question answers from cache in <1 ms.

### 2.5 Sync engine — edge ⇄ cloud

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

### 2.6 Data renewal

Memory rots. Renewal is a first-class subsystem, not a cron job.

- **Freshness scoring** — every point carries `ttl`, `confidence`, `last_verified_at`. A decay function marks points `stale` before they mislead anyone.
- **Re-embedding on model upgrade** — when the embedder version bumps, a **dual-space migration** begins: both spaces are queried and results fused while a background job re-embeds the corpus in priority order (hot tier first). Zero read downtime, zero mixed-space corruption.
- **Progressive renewal** — re-embedding is checkpointed and interruptible; power loss resumes from the last checkpoint.
- **Source revalidation** — points linked to an external source are re-fetched and diffed when connectivity allows; changed sources supersede the old memory and keep the audit chain.
- **Compaction & decay** — expired points drop to cold, then to tombstone. Pinned and high-salience points are exempt.
- **Shadow evaluation** — before a renewed model is promoted, a golden query set is replayed against both spaces; promotion is blocked on recall regression.

### 2.7 Policy & privacy engine

- On-device **PII / sensitivity classifier** (ONNX) tags every chunk `public | internal | sensitive | restricted` before it is ever written.
- Declarative policy (`YAML`, hot-reloadable): *restricted never leaves the device; sensitive syncs only redacted; public syncs freely.*
- **Redaction pipeline** — named-entity masking with a reversible local-only vault so the device can still resolve what the cloud can never see.
- Encryption at rest (AES-256-GCM, key in TPM/Secure Enclave/keyring), mTLS in flight, per-device identity certificates.
- **Tamper-evident audit log** — hash-chained, append-only record of every read, sync and escalation.

### 2.8 Instantaneous reconnection

The part most projects hand-wave. Reconnection is sub-second and stateful.

- **Connectivity oracle** — active probes + OS network-change hooks + RTT/jitter/loss EWMA classify the link as `OFFLINE / DEGRADED / METERED / HEALTHY`. State changes fire in milliseconds, not on the next poll tick.
- **Pre-warmed transport** — QUIC/HTTP3 with **0-RTT session resumption** and a warm connection pool; reconnect skips the full handshake.
- **Session continuation tokens** — the server remembers the device's sync cursor, so resumption is `"continue from op 84,213"`, not a re-handshake.
- **Operation queue** — every action taken offline is durably queued, idempotency-keyed, and replayed in causal order on reconnect. Nothing is lost, nothing is applied twice.
- **Hedged requests** — during `DEGRADED`, duplicate requests are raced across paths and the first response wins.
- **Reconnect storm control** — decorrelated jitter backoff + fleet-wide token bucket so 10,000 devices returning at once don't DDoS the coordinator.
- **Circuit breaker** — per-endpoint breakers trip fast and half-open probe, so a sick cloud endpoint never stalls the local path.
- **Optimistic UI contract** — the WebSocket pushes `link_state` transitions so the frontend flips between LOCAL and FUSED modes the moment the link moves.

### 2.9 Realtime & transport layer

- **WebSocket multiplex** — one socket, logical channels (`telemetry`, `sync`, `search`, `reasoning_trace`, `alerts`), heartbeats with server-side liveness detection.
- **Event bus** — internal pub/sub; every subsystem emits structured events which the gateway fans out to subscribed UIs.
- **Server-Sent Events fallback** for locked-down networks; **gRPC** for device↔cloud; **REST** for control plane.
- **Backpressure-aware streaming** — slow consumers get sampled, not buffered to death.

### 2.10 Reliability & operations

- **Supervisor** with per-subsystem health, restart budgets and crash-loop detection.
- **WAL + crash recovery** — an unclean shutdown replays the write-ahead log; a half-written batch is never half-visible.
- **Chaos hooks** — inject link loss, packet loss, clock skew, disk-full and process kill from an admin endpoint. The demo can *prove* resilience live, on stage.
- **Observability** — OpenTelemetry traces spanning `edge → link → Triton → back`, Prometheus metrics, structured logs; p50/p95/p99 exposed to the UI.
- **Benchmark harness** — reproducible recall@k, latency and sync-convergence numbers, checked in.

---

## 3. Planned stack

| Layer | Choice |
|---|---|
| Edge runtime | Python 3.11 · FastAPI · Uvicorn · asyncio |
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
| `WS` | `/api/v1/stream` | Multiplexed live telemetry, sync events, reasoning traces |

---

## 5. Frontend

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

## 6. Status

- [x] Problem statement locked — PS03
- [x] Backend architecture & feature plan
- [x] Frontend shell + live backend contract
- [ ] Qdrant Edge memory core
- [ ] ONNX embed / rerank pipeline
- [ ] Sync engine + conflict arbiter
- [ ] Triton escalation tier
- [ ] Renewal orchestrator
- [ ] Chaos harness + benchmarks

---

*Code Cubicle 6.0 · 3 OCT online · 11 OCT offline*
