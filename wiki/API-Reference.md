# API Reference

Full interactive surface at `http://localhost:8000/docs` once the node is
running. This page is the map.

## Everyday

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/memory/ingest` | Push a memory through the ingest pipeline |
| `POST` | `/api/v1/search` | `{query, k, filters, mode}` → hybrid results + latency breakdown + trace |
| `POST` | `/api/v1/ask` | Agentic answer: plan → retrieve → verify → cited answer |
| `GET` | `/api/v1/health` | Liveness, ONNX execution provider, active vector backend, uptime |
| `GET` | `/api/v1/memory/stats` | Point counts per tier and collection, quantization, disk, renewal progress |

## Qdrant

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/qdrant` | The engine: named vectors, collection schemas, which payload indexes are actually live, the routing policy with both paths' measured p95, and the plan the last query executed |
| `POST` | `/api/v1/qdrant/bakeoff` | Run one query down both paths and report the difference: overlap, rank agreement, each path's latency |
| `GET` | `/api/v1/qdrant/facets` | Payload value counts computed by the engine rather than by a scan in this process |
| `GET` | `/api/v1/qdrant/map` | The corpus as Qdrant holds it, projected to two dimensions, with the share of variance those two axes actually carry |
| `GET` | `/api/v1/index` | Index strategies, calibration, planner statistics |

## Sync and mesh

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/sync/status` | Cursor, pending ops, divergence, last convergence, conflicts |
| `POST` | `/api/v1/sync/trigger` | Force a reconciliation pass |
| `GET` | `/api/v1/sync/egress` | The queue priced against the link that exists right now — what would go, in what order, why, and what the same bytes would have carried in write order. **Sends nothing** |
| `GET` | `/api/v1/sync/conflicts` · `POST /conflicts/review` | Conflict records and the human review queue |
| `GET` | `/api/v1/mesh/status` · `POST /mesh/round` | Peer membership and anti-entropy. Carries this device's public key, the devices it has learned keys for, and running counts of operations verified, refused as forged, and refused by policy |
| `POST` | `/api/v1/mesh/exchange` | The receiving half of the mesh, when the peer is another device |
| `POST` | `/api/v1/mesh/offline` | Pull this device's radio, or put it back |
| `POST` | `/api/v1/mesh/attack` | Mount one of four operations — honest, tampered, impersonated, unsigned — through the same handler a peer reaches, and report what it did with it. The honest case is the control |

## Integrity and operations

| Method | Path | Purpose |
|---|---|---|
| `GET/POST` | `/api/v1/integrity/*` | fsck, scrub, archive, generations, point-in-time restore |
| `GET` | `/api/v1/integrity/invariants` · `POST /invariants/check` | What the node has asserted about itself while running, per-invariant counters, and the scope each local check can actually establish |
| `GET` | `/api/v1/integrity/recovery` | What the last boot recovered, and what it could not |
| `GET/POST` | `/api/v1/slo` · `/slo/override` | Error budget, degradation level, manual pin |
| `POST` | `/api/v1/chaos/{fault}` | Inject a fault (demo and testing only) |
| `POST` | `/api/v1/chaos/kill` | Send this node SIGKILL; refused without a supervisor |
| `GET` | `/api/v1/energy` | Joules per operation, answers per 1% of battery |
| `GET` | `/api/v1/metrics` | Prometheus exposition (`/metrics/json` for the raw snapshot) |
| `GET` | `/api/v1/traces` | Recent query span trees and their hotspots |
| `GET` | `/api/v1/scheduler` | QoS lane depths, deadline misses, pressure |

## Knowledge, learning and provenance

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/graph/stats` · `/entities` · `POST /paths` | Knowledge graph structure and multi-hop reasoning |
| `GET` | `/api/v1/graph/as-of` · `/diff` | Time travel: what was believed, and what changed |
| `POST` | `/api/v1/learning/feedback` | Teach the on-device adapter from a real choice |
| `POST` | `/api/v1/learning/round` | One secure-aggregation round |
| `GET` | `/api/v1/learning/confidence` | Conformal calibration and realised coverage |
| `GET` | `/api/v1/space` · `/space/anisotropy` | The embedding space: pooling, fitted geometry, lexicon, gate history, measured conditioning |
| `POST` | `/api/v1/space/evaluate` · `/space/arm` | Score candidate spaces against the shipped one; arming is refused while any vector is still in the old space |
| `GET` | `/api/v1/audit` | Hash-chained audit entries plus chain verification |
| `GET` | `/api/v1/provenance` · `POST /provenance/verify` | The Merkle receipt over model graphs and configuration, and a file-by-file comparison |
| `GET/POST` | `/api/v1/tenants/*` | Tenants, scoped API keys, quotas |
| `WS` | `/api/v1/stream` | Multiplexed live telemetry, sync events, reasoning traces |

## What a search response carries

```jsonc
{
  "results": [ { "id": …, "text": …, "score": …,
                 "matched_by": ["dense", "sparse"],   // which half found it
                 "breakdown": { … } } ],             // every scoring term
  "stages":  { "understand_ms": …, "embed_ms": …,
               "engine_ms": … },                     // or dense/sparse/fusion
  "plan":    { "plan": "pre_filter", "reason": …,
               "engine": { "calls": 1, "queries_in_call": 3,
                           "route": { "path": "engine", "reason": … } } },
  "confidence": { "guarantee": … },                  // conformal, calibrated
  "diversity":  { … },
  "degradation": { "level": …, "shed": […] },        // what it was allowed to use
  "trace":   { "name": "search", "children": […] }   // span tree
}
```

An answer that cannot be argued with is not an answer. Every one of those fields
exists so a result can be challenged.
