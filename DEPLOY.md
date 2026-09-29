# Deploying AegisEdge

**Console on Netlify, node on Render.** The split is not a preference — it is
what the two halves actually need.

| | What it is | Where it goes | Why |
|---|---|---|---|
| `frontend/` | A static page, no build step | **Netlify** | Nothing to run. Serve the files. |
| `backend/` | A stateful Python process | **Render** | Embedded Qdrant on disk, two ONNX graphs compiled at boot, an fsynced write-ahead log, a WebSocket, and background sync, gossip and invariant loops. It needs a long-lived process and a persistent disk. |

Netlify serves static assets and short-lived JS/TS edge functions. It has no
Python function runtime, no persistent disk and no long-running process, so the
node cannot go there. That is the whole reason for two services.

Nothing in `frontend/` or `backend/` changes for any of this. The console
already reads `window.AEGIS_API`, and an edge function supplies it.

---

## 1. The node, on Render

**New → Blueprint → point it at this repository.** `render.yaml` describes the
service; Render reads it.

What it sets up:

- Python 3.11, `pip install -r backend/requirements.txt`
- `uvicorn aegis.main:app --host 0.0.0.0 --port $PORT`
- A **10 GB persistent disk at `/var/aegis`**, with `AEGIS_DATA_DIR` and
  `AEGIS_MODEL_DIR` pointing into it
- Health check on `/api/v1/health`
- **One instance**, deliberately

Then set one environment variable in the Render dashboard once the Netlify site
exists:

```
AEGIS_CORS_ORIGINS = https://your-site.netlify.app
```

### What to expect on first boot

It is slow, once. The node compiles `embedder.onnx` and `reranker.onnx` from the
weights bundled in the wheel and content-addresses them into the registry —
about **127 MB** written to the disk. Because the model directory is on the
persistent disk, later deploys skip it.

Nothing is downloaded during this. Provisioning never touches the network.

### Why one instance

Embedded Qdrant takes an exclusive lock on its directory and the write-ahead log
has a single writer. A second instance on the same disk is refused at boot —
correctly, but the deploy would look broken.

Scale by running **more nodes with their own disks** and letting them find each
other over the mesh. That is what the sync engine is for, and it is the shape
the whole system was designed around.

### A paid instance is required

The free tier has no persistent disk and spins down when idle. Without a disk
the node recompiles the models on every wake and loses its memories — which is
not a deployment, it is a demo that resets.

---

## 2. The console, on Netlify

**Add new site → import this repository.** `netlify.toml` does the rest:
publish `frontend/`, no build command, revalidating cache headers.

Set one environment variable in **Site configuration → Environment variables**:

```
AEGIS_API_URL = https://aegisedge-node.onrender.com
```

`netlify/edge-functions/aegis-config.ts` injects that into the document head
ahead of every console script, because `app.js` reads `window.AEGIS_API` once at
module scope. If the variable is unset nothing is injected and the console falls
back to `http://localhost:8000`, which is the right behaviour for a preview
deploy with no node behind it: it reports the node as unreachable and shows
nothing, rather than inventing values.

---

## 3. Check it

```bash
curl https://aegisedge-node.onrender.com/api/v1/health
```

Then open the Netlify site. The boot screen reads the node's own `/health` while
it plays, so if the execution provider, embedder and vector backend fill in, the
two halves are talking.

`INSPECT IT` should show live cards, `QDRANT · QUERY ENGINE` should draw a plan
after a search, and `EVENT STREAM` should fill as things happen — that last one
is the WebSocket, and it is the reason for the direct-connection setup below.

### What the browser console will show, and why it is fine

Two harmless errors, worth knowing so they are not mistaken for a broken deploy:

- **`localhost:8201` and `localhost:8202`, connection refused.** The `USE IT`
  mode has a two-device mesh demonstration that talks to two *other* nodes at
  those fixed local addresses. It is a local-only demo — run three nodes on one
  machine and it works; deployed, those two devices do not exist. Every other
  mode is unaffected, and the single deployed node is fully functional,
  including its own mesh endpoints.
- **`/favicon.ico`, 404.** There is no favicon in the repository.

Everything else was checked against a deployed-shaped setup — console on one
origin, node on another, talking over CORS: all fifteen console panels populate,
search returns ranked results with their score breakdown, the Qdrant panel draws
the plan the engine executed, the corpus map plots, and the event stream fills
over the WebSocket.

---

## Direct connection, not a proxy

The console talks to the node's own URL. Cross-origin, allowed by
`AEGIS_CORS_ORIGINS`.

Netlify can proxy `/api/*` to the node instead, and `netlify.toml` carries that
block commented out. It works for every REST call — and **Netlify does not proxy
WebSockets**, so the live event stream would stay empty while everything else
worked. Direct is the default because it is the one where every feature works.

The trade is that the node's URL is visible in the page. If that matters more
than the event stream, uncomment the redirect block and leave `AEGIS_API_URL`
unset so the console calls its own origin.

---

## Locking it down

The defaults are set for a demo someone can open and use. For anything with real
data in it:

| Variable | Set it to |
|---|---|
| `AEGIS_CORS_ORIGINS` | The Netlify site's exact URL, never `*` |
| `AEGIS_REQUIRE_AUTH` | `1`, then issue scoped keys at `/api/v1/tenants/*` |
| `AEGIS_ROOT_PUBLIC_KEY` | A fleet root key, so peers must present a signed certificate instead of being trusted on first use |

With `AEGIS_REQUIRE_AUTH=1` every route needs a key, including the ones the
console calls — so the console needs one too, and that is a deliberate wall
rather than an oversight.

---

## Pointing at a Qdrant Server

Set `AEGIS_QDRANT_URL` (and `AEGIS_QDRANT_API_KEY`) and the node moves its query
path onto a compiled engine. The router will then choose it for every query,
because on a server one `query_points` call beats four round trips — the
embedded client's local mode is the only reason it ever picks the local index.

Nothing else changes. The same adapter, the same code path, one setting.
