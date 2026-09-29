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
- The ONNX graphs compiled **at build time**, so a cold start does not pay for
  them
- `uvicorn aegis.main:app --host 0.0.0.0 --port $PORT`
- Health check on `/api/v1/health`
- **One instance**, deliberately
- The **free** plan — no card, no disk

Then set one environment variable in the Render dashboard once the Netlify site
exists:

```
AEGIS_CORS_ORIGINS = https://your-site.netlify.app
```

### What the free tier costs you

Two real limitations, neither hidden:

**The filesystem is ephemeral.** A persistent disk needs a paid instance, so
memories, the write-ahead log and the Qdrant collections are gone when the
container restarts. The node works completely; it just starts empty again.

**It sleeps after 15 minutes with no requests.** The first request after that
wakes it.

Those two combine into something worth stating plainly, because it is easy to
read the first one as worse than it is: **the container is what holds the
memories, and closing the page does not restart the container.** What restarts
it is fifteen minutes of no traffic at all, a redeploy, or Render cycling the
instance.

So a keep-warm ping — anything hitting `/api/v1/health` every ten minutes —
buys the instant first click *and* memories that stay put, because the idle
timer never expires. Render can still cycle a free instance on its own and a
push to `main` still redeploys, so this is reliable rather than guaranteed. The
guarantee is the disk.

The wake is fast because the models are compiled during the build rather than on
boot: measured **3 seconds** from process start to a healthy response, against
about 45 seconds when the graphs were built on first use. Nothing is downloaded
at any point — the weights ship in the wheel and provisioning never touches the
network.

The node needs about **270 MB** of RSS with a small corpus, against the free
tier's 512 MB, so there is room but not unlimited room: a corpus in the tens of
thousands will want the paid instance.

### Upgrading later

Change `plan: free` to `plan: starter` and put the disk block back:

```yaml
    disk:
      name: aegis-data
      mountPath: /var/aegis
      sizeGB: 10
```

...then point `AEGIS_DATA_DIR` and `AEGIS_MODEL_DIR` at `/var/aegis/...`.
Nothing else changes, and memories survive restarts from then on.

### Why one instance

Embedded Qdrant takes an exclusive lock on its directory and the write-ahead log
has a single writer. A second instance on the same disk is refused at boot —
correctly, but the deploy would look broken.

Scale by running **more nodes with their own disks** and letting them find each
other over the mesh. That is what the sync engine is for, and it is the shape
the whole system was designed around.

### The second device, and why identity has to be pinned

`render.yaml` describes **two** services. The second one, `aegisedge-node-b`,
exists because the console's `USE IT` mode demonstrates two devices reconciling
directly with no cloud between them — the part of the problem statement that is
easiest to claim and hardest to show. It needs two node processes.

They peer over HTTP, which is why both carry `AEGIS_MESH_TRANSPORT=http`. With
the in-process default they could not see each other at all.

**Both need `AEGIS_DEVICE_KEY_PEM` set**, and that is not optional here. The
node keeps its Ed25519 key at `<data_dir>/device_key.pem` and generates one when
it is missing — so on an ephemeral filesystem it gets a *new identity on every
restart*. A peer that already learned the old key then refuses everything the
node sends as a forgery. That is trust-on-first-use working exactly as designed,
and it is also a mesh that silently stops converging after the first sleep.

Measured, because it is the kind of thing that is easy to assume either way:
unpinned, two boots from a wiped directory produced two different public keys;
pinned, they produced the same one. The start command writes the PEM at mode
`0600` before the node boots, and skips the whole thing when the variable is
unset — which is the right behaviour for a single node with no peers.

Generate a keypair per device with:

```bash
python3 -c "
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
print(Ed25519PrivateKey.generate().private_bytes(
    serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
    serialization.NoEncryption()).decode())"
```

Paste each into that service's `AEGIS_DEVICE_KEY_PEM` in the Render dashboard
and mark it secret.

---

## 2. The console, on Netlify

**Add new site → import this repository.** `netlify.toml` does the rest:
publish `frontend/`, no build command, revalidating cache headers.

Set these in **Site configuration → Environment variables**:

| Key | Value | Why |
|---|---|---|
| `AEGIS_API_URL` | `https://aegisedge-node.onrender.com` | The node the console reads |
| `AEGIS_DEVICE_A_URL` | `https://aegisedge-node.onrender.com` | `USE IT`'s first device |
| `AEGIS_DEVICE_B_URL` | `https://aegisedge-node-b.onrender.com` | `USE IT`'s second device |

None of them may be marked **secret** — a secret value is not readable at
request time, so the edge function would get nothing and the console would sit
there reporting the node unreachable.

The two device URLs are read together: set both and the deployed console points
`USE IT` at the deployed pair, set neither and it keeps its local defaults
rather than being handed a half-configured mesh.

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

- **`/favicon.ico`, 404.** There is no favicon in the repository.
- **`localhost:8201` / `localhost:8202`, connection refused** — only if you did
  *not* set `AEGIS_DEVICE_A_URL` and `AEGIS_DEVICE_B_URL`. Those are `USE IT`'s
  local development defaults. Set both and it points at the deployed pair.

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
