# AegisEdge — backend

The edge node. FastAPI on the outside, an offline-first memory and sync
engine on the inside.

```bash
pip install -r requirements.txt
uvicorn aegis.main:app --reload --port 8000     # API + WebSocket
python3 scripts/demo.py                          # full lifecycle, no server
python3 -m pytest tests -q                       # 52 tests
```

Open `http://localhost:8000/docs` for the live OpenAPI surface, or point the
frontend at it (it defaults to `http://localhost:8000`).

## Module map

| Path | What lives there |
|---|---|
| `aegis/node.py` | Composition root — every subsystem is built and supervised here |
| `aegis/config.py` | All configuration, env-overridable (`AEGIS_*`) |
| `aegis/core/` | HLC clock, event bus, supervisor, metrics, breaker, backoff, token bucket |
| `aegis/memory/` | Schema, WAL, quantizers, tiered index, Qdrant Edge adapter, compactor, consolidation |
| `aegis/inference/` | ONNX session + EP ladder, micro-batcher, embedder, sparse encoder, reranker, classifier, thermal governor, model registry, Triton client |
| `aegis/retrieval/` | RRF fusion, scoring, semantic cache, contradiction detection, pipeline, agent |
| `aegis/sync/` | CRDT op log, Merkle digests, durable queue, connectivity oracle, transports, conflict arbiter, engine |
| `aegis/renewal/` | Freshness sweeps, dual-space migrator, scheduler |
| `aegis/policy/` | Policy engine, redaction vault, hash-chained audit log |
| `aegis/chaos/` | Fault injection |
| `aegis/api/` | Routers, schemas, WebSocket gateway |

## Optional dependencies

The node runs with none of these and reports exactly which path it is on
(`/api/v1/health` → `memory_backend`, `execution_provider`, `fallback_encoder`):

| Package | Enables | Without it |
|---|---|---|
| `qdrant-client` | Qdrant Edge embedded collections | `native-tiered` NumPy store, same semantics |
| `onnxruntime` | real ONNX graphs + accelerators | deterministic hashed-n-gram encoder |
| `tritonclient[grpc]` | cloud escalation tier | escalation declines, everything stays local |
| `psutil` | real thermal/battery telemetry | synthesised sensor drift |

Drop graphs into `models/<name>.<variant>.onnx` and they are picked up on the
next boot — the registry verifies each digest before loading it.

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
