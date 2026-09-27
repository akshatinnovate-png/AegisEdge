# Configuration

Everything is environment-overridable with an `AEGIS_` prefix. Defaults are in
`backend/aegis/config.py`.

## The ones worth knowing

| Variable | Effect |
|---|---|
| `AEGIS_DATA_DIR` | One node per directory — embedded Qdrant is single-writer, and a second node on the same directory gets a message that says so rather than a `BlockingIOError` from three libraries down |
| `AEGIS_QDRANT_URL` | Point the store at a Qdrant Server instead of the embedded instance |
| `AEGIS_QDRANT_API_KEY` | Credential for the above |
| `AEGIS_QDRANT_QUERY_PATH` | `auto` (route on measured latency), `engine` or `index` to pin one path |
| `AEGIS_QDRANT_LATE_TOKENS` | Tokens kept per memory for engine-side MaxSim. `0` (default) stores no late vector |
| `AEGIS_REQUIRE_QDRANT=0` | Explicitly allow the internal store — it will say so in `/health` |
| `AEGIS_CLOUD_URL` | The sync coordinator: a Qdrant URL, or empty for a real embedded Qdrant under `<data_dir>/cloud` |
| `AEGIS_REQUIRE_AUTH=1` | Enforce API keys and tenant resolution on every route |
| `AEGIS_ROOT_PUBLIC_KEY` | Require every peer to present a certificate signed by the fleet root; empty means trust-on-first-use |
| `AEGIS_MESH_TRANSPORT` | `memory` (in-process peers, for tests and simulated fleets) or `http` (other node processes) |
| `AEGIS_NODE_ID` | This device's identity in the fleet |
| `AEGIS_NODE_ENDPOINT` | Where peers reach this node over HTTP |

## Why Qdrant is required by default

A node that silently falls back to an internal store while claiming to be
Qdrant-backed is telling its operator something untrue about where their data
lives. So it is a hard failure, and `AEGIS_REQUIRE_QDRANT=0` is the explicit
opt-out. The active backend is reported verbatim in `/health` as `qdrant-local`
or `qdrant-server`.

## Dependencies are not optional

| Package | Provides |
|---|---|
| `wordllama`, `safetensors` | The pretrained 32000 × 256 token embedding table and its 32k BPE tokenizer, shipped in the wheel — provisioning never touches the network |
| `onnx`, `onnxruntime`, `tokenizers` | The two graphs compiled from those weights at first boot, and the runtime that executes them |
| `qdrant-client` | Qdrant, embedded by default or a server via `AEGIS_QDRANT_URL` |
| `cryptography` | Ed25519 device identity and operation signing |
| `psutil` | Platform thermal and battery telemetry; where a platform exposes none, the reading is reported **unavailable** rather than synthesised |

Only `tritonclient[grpc]` is optional, because it needs a reachable GPU server.

`tests/test_packaging.py` asserts that every third-party import is declared —
added after CI on a clean machine found `cryptography` undeclared.

## Tuning the retrieval path

| Variable | Effect |
|---|---|
| `AEGIS_DIM` | Embedding width. Set by the model bundle at boot; the pretrained table is 256-dimensional |
| `AEGIS_SPARSE_DIM` | Hashed term space for the sparse encoder (default 2¹⁸) |
| `AEGIS_HOT_CAPACITY` / `AEGIS_WARM_CAPACITY` | Tier sizes before points spill to the memmapped cold tier |
| `AEGIS_MAX_TOKENS` | Token budget per document for the encoder |
| `AEGIS_BATCH_WINDOW_MS` / `AEGIS_MAX_BATCH` | Micro-batching for concurrent ingest and query |
| `AEGIS_THERMAL_CEILING_C` / `AEGIS_BATTERY_FLOOR_PCT` | Where the governor starts shedding inference work |
