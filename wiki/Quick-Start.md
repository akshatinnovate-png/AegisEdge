# Quick Start

Two terminals. No Docker, no build step, no API keys, no network calls in the
query path.

## The node

```bash
cd backend
pip install -r requirements.txt
uvicorn aegis.main:app --port 8000
```

First boot compiles `models/embedder.onnx` and `models/reranker.onnx` from the
weights bundled in the wheel, writes `models/tokenizer.json` and
`models/provenance.json`, and content-addresses both graphs into the registry.
**Nothing is downloaded.** Later boots load the serialized optimized graph.

The node refuses to start without real weights or a real vector store. There is
no fallback encoder and no internal store masquerading as Qdrant — see
[Configuration](Configuration) if you want to override that explicitly.

## The console

```bash
cd frontend
python3 -m http.server 5173
# open http://localhost:5173
```

It defaults to `http://localhost:8000`. When the node is unreachable the console
says so and shows nothing — a dashboard that invents plausible values is worse
than a blank one, because you cannot tell the difference until it matters.

## Prove it to yourself

```bash
cd backend

python3 -m pytest tests -q                  # 385 tests

python3 scripts/simulate.py --seeds 40      # signed: expect no counterexample
python3 scripts/simulate.py --seeds 40 --unsigned   # control: expect plenty

python3 scripts/qdrant_bakeoff.py           # the engine's path vs this node's index
python3 scripts/egress_bakeoff.py           # value-first vs write order on a dying link
python3 scripts/strategy_bakeoff.py         # flat vs HNSW vs IVF-PQ on one corpus

python3 scripts/audit_claims.py             # every documented number vs its artefact
python3 scripts/audit_determinism.py        # no simulated module may read the clock
```

The unsigned control is the one to run if you only run one. It is the run that
makes the clean one mean something.

## No server at all

```bash
python3 scripts/demo.py     # the whole lifecycle in-process
```
