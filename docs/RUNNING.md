# Running it

*Part of [AegisEdge](../README.md).*

---

## Quick start — run it, and use every mode

Once installed, it is two commands to a running system and nothing needs a
network after that. Every command below was run to write this section, and the
paths are all relative to the repository root.

### 1. Install

```bash
git clone https://github.com/akshatinnovate-png/AegisEdge.git
cd AegisEdge
pip install -r backend/requirements.txt          # add -dev for the tests
```

That pulls the pretrained weights and Qdrant with it. **Nothing is downloaded
at run time** — the ONNX graphs are compiled from those weights on first boot.

### 2. Start the node, and the console

Two terminals.

```bash
# terminal 1 — the node
cd backend
uvicorn aegis.main:app --port 8000
```

```bash
# terminal 2 — the console
cd frontend
python3 -m http.server 5173
```

Open **http://localhost:5173**. The console finds the node on `:8000` by
itself. If the header says `NO NODE`, the node is not up — every figure blanks
rather than showing you a stale number.

### 3. The three modes

They are the three buttons in the header.

| Button | What it is | Try this |
|---|---|---|
| **USE IT** | The product. Capture a memory, ask a question, get a cited answer. | Type something into **capture**, then ask a question about it. The right-hand rail shows what the node did with it — the tier, the handling class, the policy rule that decided whether it may ever leave the device. |
| **PROVE IT** | Six claims, each with the button that would falsify it. | Press **KILL THIS NODE · SIGKILL** and watch it come back with the same memory count. Then drag the load dial until the degradation ladder sheds a stage. |
| **INSPECT IT** | The raw machinery — live store, index, mesh, sync, traces. | Search for something and read the per-stage timings: understand, embed, plan, dense, sparse, fuse, rerank. |

The node **ships empty**. Nothing is seeded, so whatever you see is something
you put there.

### 4. Feed it without the browser

```bash
curl -X POST localhost:8000/api/v1/memory/ingest \
  -H 'content-type: application/json' \
  -d '{"text":"conveyor 3 vibration spike on the night shift"}'

curl -X POST localhost:8000/api/v1/ask \
  -H 'content-type: application/json' \
  -d '{"query":"what happened on the conveyor?"}'
```

The answer comes back with a citation pointing at the memory it came from, and
a `trace` giving every stage its own millisecond figure.

### 5. Two real devices, gossiping with no cloud

```bash
cd backend
AEGIS_DATA_DIR=/tmp/devA AEGIS_NODE_ID=device-A AEGIS_MESH_TRANSPORT=http \
  uvicorn aegis.main:app --port 8201 &
AEGIS_DATA_DIR=/tmp/devB AEGIS_NODE_ID=device-B AEGIS_MESH_TRANSPORT=http \
  uvicorn aegis.main:app --port 8202 &
cd ../frontend && python3 -m http.server 5173
```

Open the console, choose **USE IT**, press **PAIR DEVICES**. Save a memory on
one, pull its radio, save more, put the radio back — and watch anti-entropy
move exactly what the other side was missing.

### 6. The whole system in one command

No server, no browser, nothing to click:

```bash
cd backend && python3 scripts/demo.py
```

Cold boot and WAL replay, retrieval with the link down, policy blocking a
restricted memory, an outage queued and replayed, a contradiction superseded,
a re-embedding migration, four injected faults, and a peer-to-peer mesh round
with the uplink down.

### 7. Check the claims yourself

```bash
cd backend
python3 -m pytest -q                           # 319 tests
python3 scripts/audit_determinism.py           # the determinism lint
python3 scripts/simulate.py --seeds 200        # the simulator, ~2 min
python3 scripts/simulate.py --seeds 200 --unsigned --no-shrink \
        --stop-after 200                       # the control, ~20 s
python3 scripts/bench.py                       # index recall and latency
python3 scripts/repro_growth.py                # the one open defect, in two arms
python3 scripts/scale_probe.py --to 10000      # the cost of a memory as the corpus grows
python3 scripts/memory_breakdown.py            # ...and where that cost goes
```

The node also checks the simulator's invariants against itself while it runs.
`curl localhost:8000/api/v1/integrity/invariants` says how many assertions it
has made and how many did not hold.

Those two simulator runs are the pair worth doing together — same simulator,
same seeds, signing on and off:

```
  --seeds 200                        invariant failures   0 of 200
  --seeds 200 --unsigned             invariant failures   165 of 200
```

Slower, if you want the full picture: `scripts/stress.py --phases all --out ../testlogs`
(~21 min), and `scripts/simulate.py --seeds 3000 --steps 400 --peers 6` (~74 min,
and the run committed in `testlogs/simulation.json`).

---

## 10. Running it

```bash
# backend — the node
cd backend
pip install -r requirements.txt      # includes the pretrained weights and Qdrant
uvicorn aegis.main:app --port 8000      # REST + WebSocket on :8000
python3 scripts/supervise.py --port 8000   # ...or supervised, so it can be killed
python3 scripts/demo.py                 # whole lifecycle in one process, no server
python3 scripts/bench.py                # index recall + latency, measured here
python3 -m pytest tests -q              # 247 tests

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

