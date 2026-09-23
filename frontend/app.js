/* ============================================================
   AegisEdge — console frontend
   Talks REST + WebSocket to the edge node. If no node answers,
   it runs a local simulation instead of showing a dead page,
   which is the whole premise of an offline-first product.
   ============================================================ */

const API = (window.AEGIS_API || "http://localhost:8000").replace(/\/$/, "");
const WS_URL = API.replace(/^http/, "ws") + "/api/v1/stream";

const $ = (id) => document.getElementById(id);

const state = {
  live: false,          // is a real backend answering?
  link: "offline",      // offline | degraded | healthy
  socket: null,
  retries: 0,
  mem: { hot: 0, warm: 0, cold: 0 },
  sync: { queued: 0, divergent: 0, conflicts: 0, progress: 0, state: "HOLDING" },
  renewal: { state: "IDLE", done: 0, total: 0, stale: 0, progress: 0 },
  escalations: 0,
};

/* ── boot sequence ─────────────────────────────────────── */

const POST_LINES = [
  ["AegisEdge Node Firmware  v0.1.4-edge", "hi"],
  ["Copyright (C) Code Cubicle 6.0 — PS03", ""],
  ["", ""],
  ["Detecting compute...................... ", "ok", "OK"],
  ["Probing execution providers............ ", "ok", "CPU/XNNPACK"],
  ["Loading onnx://embedder/bge-small-int8. ", "ok", "OK"],
  ["Loading onnx://reranker/cross-enc-int8. ", "ok", "OK"],
  ["Loading onnx://classifier/sensitivity.. ", "ok", "OK"],
  ["Mounting qdrant-edge (embedded)........ ", "ok", "OK"],
  ["  collections: episodic semantic procedural sensor", ""],
  ["Replaying write-ahead log.............. ", "ok", "0 TORN"],
  ["Restoring sync cursor.................. ", "ok", "RESUMED"],
  ["Opening telemetry bus.................. ", "ok", "OK"],
  ["Probing uplink......................... ", "ok", "DEFERRED"],
  ["", ""],
  ["Local memory is authoritative. Network optional.", "hi"],
];

const CAPTIONS = [
  "mounting local vector memory…",
  "warming onnx sessions…",
  "replaying write-ahead log…",
  "restoring sync cursor…",
  "haze clearing…",
];

function runBoot() {
  const post = $("post");
  const caption = $("bootCaption");
  let i = 0;

  const tick = () => {
    if (i < POST_LINES.length) {
      const [text, cls, tail] = POST_LINES[i];
      const row = document.createElement("div");
      row.textContent = text;
      if (cls === "hi") row.className = "hi";
      if (tail) {
        const s = document.createElement("span");
        s.className = "ok";
        s.textContent = "[ " + tail + " ]";
        row.appendChild(s);
      }
      post.appendChild(row);
      caption.textContent = CAPTIONS[Math.min(CAPTIONS.length - 1, Math.floor(i / 3.4))];
      i++;
      setTimeout(tick, 60 + Math.random() * 110);
    } else {
      setTimeout(finishBoot, 520);
    }
  };
  setTimeout(tick, 320);
}

function finishBoot() {
  $("bootFlash").classList.add("is-on");
  setTimeout(() => {
    $("boot").classList.add("is-done");
    $("page").classList.add("is-in");
  }, 190);
  setTimeout(start, 700);
}

/* ── backend client ────────────────────────────────────── */

async function get(path, ms = 2200) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), ms);
  try {
    const res = await fetch(API + path, { signal: ctrl.signal });
    if (!res.ok) throw new Error(res.status);
    return await res.json();
  } finally {
    clearTimeout(t);
  }
}

async function post(path, body, ms = 6000) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), ms);
  try {
    const res = await fetch(API + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
      signal: ctrl.signal,
    });
    if (!res.ok) throw new Error(res.status);
    return await res.json();
  } finally {
    clearTimeout(t);
  }
}

async function probe() {
  const t0 = performance.now();
  try {
    const h = await get("/api/v1/health");
    const rtt = Math.round(performance.now() - t0);
    if (!state.live) log("link", "node reachable — switching to <b>FUSED</b> mode", "ok");
    state.live = true;
    state.retries = 0;
    setLink(rtt > 220 ? "degraded" : "healthy", rtt);
    $("dataSrc").textContent = "SOURCE · LIVE NODE";
    if (h.execution_provider) $("epName").textContent = h.execution_provider;
    if (h.node_id) $("nodeId").textContent = h.node_id;
    await pullAll();
    openSocket();
  } catch {
    if (state.live) log("link", "uplink lost — <b>local memory still serving</b>", "warn");
    state.live = false;
    setLink("offline", null);
    $("dataSrc").textContent = "SOURCE · SIMULATED";
    state.retries++;
  }
}

async function pullAll() {
  const [mem, sync, ren, node] = await Promise.allSettled([
    get("/api/v1/memory/stats"),
    get("/api/v1/sync/status"),
    get("/api/v1/renewal/status"),
    get("/api/v1/node/state"),
  ]);
  if (mem.status === "fulfilled") applyMemory(mem.value);
  if (sync.status === "fulfilled") applySync(sync.value);
  if (ren.status === "fulfilled") applyRenewal(ren.value);
  if (node.status === "fulfilled") applyNode(node.value);
}

function openSocket() {
  if (state.socket && state.socket.readyState <= 1) return;
  let ws;
  try {
    ws = new WebSocket(WS_URL);
  } catch {
    return;
  }
  state.socket = ws;
  ws.onopen = () => log("stream", "telemetry channel open", "ok");
  ws.onmessage = (e) => {
    let msg;
    try {
      msg = JSON.parse(e.data);
    } catch {
      return;
    }
    handleEvent(msg);
  };
  ws.onclose = () => {
    // decorrelated-jitter backoff, matching the node's own reconnect policy
    const delay = Math.min(8000, 300 * Math.pow(1.7, state.retries) + Math.random() * 400);
    state.retries++;
    setTimeout(() => state.live && openSocket(), delay);
  };
  ws.onerror = () => ws.close();
}

function handleEvent(msg) {
  switch (msg.channel) {
    case "telemetry":
      if (msg.memory) applyMemory(msg.memory);
      if (msg.node) applyNode(msg.node);
      break;
    case "sync":
      applySync(msg);
      break;
    case "renewal":
      applyRenewal(msg);
      break;
    case "alerts":
    default:
      if (msg.message) log(msg.kind || "node", msg.message, msg.level);
  }
}

/* ── state → DOM ───────────────────────────────────────── */

function setLink(s, rtt) {
  state.link = s;
  const pill = $("linkPill");
  pill.dataset.state = s;
  $("linkLabel").textContent = s.toUpperCase();
  $("linkRtt").textContent = rtt == null ? "—" : rtt + "ms";
}

function applyNode(n) {
  if (n.execution_provider) $("epName").textContent = n.execution_provider;
  if (n.embedder) $("embName").textContent = n.embedder;
  if (n.precision) $("precName").textContent = n.precision;
  if (n.embed_p95_ms != null) $("embP95").textContent = n.embed_p95_ms + " ms";
  if (n.escalations != null) $("escCount").textContent = n.escalations;
  if (n.node_id) $("nodeId").textContent = n.node_id;
  if (n.reconnect_ms != null) $("reconnMs").textContent = n.reconnect_ms;
}

function applyMemory(m) {
  state.mem = { hot: m.hot || 0, warm: m.warm || 0, cold: m.cold || 0 };
  const total = state.mem.hot + state.mem.warm + state.mem.cold;
  $("memTotal").textContent = total.toLocaleString();
  if (m.collections != null) $("memCollections").textContent = m.collections;
  for (const [k, id] of [["hot", "tHot"], ["warm", "tWarm"], ["cold", "tCold"]]) {
    const v = state.mem[k];
    $(id).textContent = v.toLocaleString();
    $(id).parentElement.querySelector("i").style.width =
      (total ? (v / total) * 100 : 0).toFixed(1) + "%";
  }
}

function applySync(s) {
  if (s.state) $("syncState").textContent = s.state;
  if (s.queued != null) $("syncQueued").textContent = s.queued;
  if (s.divergent != null) $("syncDiv").textContent = s.divergent;
  if (s.conflicts != null) $("syncConf").textContent = s.conflicts;
  if (s.progress != null) $("syncBar").style.width = Math.round(s.progress * 100) + "%";
  if (s.cursor != null) $("railRight").textContent = "SYNC CURSOR · OP " + s.cursor.toLocaleString();
}

function applyRenewal(r) {
  if (r.state) $("renState").textContent = r.state;
  if (r.done != null && r.total != null)
    $("renDone").textContent = r.done.toLocaleString() + " / " + r.total.toLocaleString();
  if (r.stale != null) $("renStale").textContent = r.stale.toLocaleString();
  if (r.progress != null) $("renBar").style.width = Math.round(r.progress * 100) + "%";
}

function log(channel, message, level) {
  const box = $("log");
  const row = document.createElement("div");
  if (level) row.className = level;
  const d = new Date();
  const ts = [d.getHours(), d.getMinutes(), d.getSeconds()]
    .map((n) => String(n).padStart(2, "0"))
    .join(":");
  row.innerHTML =
    `<span class="t">${ts}</span><span class="c">${channel.toUpperCase()}</span><span class="m">${message}</span>`;
  box.appendChild(row);
  while (box.children.length > 60) box.removeChild(box.firstChild);
  box.scrollTop = box.scrollHeight;
}

/* ── search ────────────────────────────────────────────── */

$("searchForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("searchInput").value.trim();
  if (!q) return;
  const results = $("results");
  results.innerHTML = "";
  $("searchMeta").textContent = "retrieving…";
  const t0 = performance.now();

  let payload = null;
  if (state.live) {
    try {
      payload = await post("/api/v1/search", { query: q, k: 5, mode: "hybrid" });
    } catch {
      log("search", "node did not answer — falling back to simulation", "warn");
    }
  }
  if (!payload) payload = simulateSearch(q);

  const ms = payload.latency_ms != null ? payload.latency_ms : Math.round(performance.now() - t0);
  $("searchMeta").textContent =
    `${payload.results.length} hits · ${ms} ms · dense+sparse · RRF · rerank` +
    (payload.escalated ? " · ESCALATED → TRITON" : " · LOCAL ONLY");

  payload.results.forEach((r, i) => {
    const li = document.createElement("li");
    li.style.animationDelay = i * 45 + "ms";
    li.innerHTML =
      `<div class="r-head"><span>${(r.collection || "episodic").toUpperCase()} · ${r.id || "pt-" + i}</span>` +
      `<span class="r-score">${(r.score ?? 0).toFixed(3)}</span></div>` +
      `<div class="r-text"></div>`;
    li.querySelector(".r-text").textContent = r.text || "";
    results.appendChild(li);
  });

  log("search", `"${q}" → ${payload.results.length} hits in ${ms} ms`);
});

$("syncBtn").addEventListener("click", async () => {
  const btn = $("syncBtn");
  btn.disabled = true;
  log("sync", "reconcile requested — exchanging merkle digests");
  if (state.live) {
    try {
      applySync(await post("/api/v1/sync/trigger", {}));
    } catch {
      log("sync", "node unreachable — ops stay queued locally", "warn");
    }
  } else {
    simulateReconcile();
  }
  setTimeout(() => (btn.disabled = false), 2600);
});

/* ── simulation (no backend present) ───────────────────── */

const SIM_CORPUS = [
  ["sensor", "Bay 3 conveyor vibration crossed 4.2 mm/s at 02:14; bearing signature matches the pre-failure cluster from March."],
  ["episodic", "Operator acknowledged the torque alarm and switched line 2 to manual feed for eleven minutes."],
  ["semantic", "Coolant pressure below 1.8 bar for over 90 seconds is treated as a hard stop condition on this cell."],
  ["procedural", "Recovery: isolate the drive, purge the line, re-home the gantry, then release the interlock in that order."],
  ["episodic", "Uplink dropped for 47 minutes during the night shift; 1,284 operations queued locally and replayed on reconnect."],
  ["semantic", "Restricted-class memories — anything tagged with operator identity — never leave this device under any policy."],
  ["sensor", "Ambient temperature climbed to 61°C; the governor swapped the embedder to its int8 variant to shed thermal load."],
  ["procedural", "Weekly consolidation distils near-duplicate episodic points into one semantic point and keeps provenance links."],
];

function simulateSearch(q) {
  const terms = q.toLowerCase().split(/\s+/).filter(Boolean);
  const scored = SIM_CORPUS.map(([collection, text], i) => {
    const hay = text.toLowerCase();
    const lexical = terms.reduce((a, t) => a + (hay.includes(t) ? 1 : 0), 0) / (terms.length || 1);
    const drift = ((Math.sin(i * 12.9898 + q.length) + 1) / 2) * 0.35;
    return { id: "pt-" + (4100 + i * 7), collection, text, score: 0.42 + lexical * 0.45 + drift * 0.3 };
  });
  scored.sort((a, b) => b.score - a.score);
  return {
    results: scored.slice(0, 5),
    latency_ms: 6 + Math.round(Math.random() * 5),
    escalated: false,
  };
}

function simulateReconcile() {
  state.sync.state = "RECONCILING";
  applySync({ state: "RECONCILING" });
  let p = 0;
  const step = () => {
    p += 0.07 + Math.random() * 0.13;
    if (p >= 1) {
      applySync({
        state: "CONVERGED",
        progress: 1,
        queued: 0,
        divergent: 0,
        cursor: 84213 + Math.floor(Math.random() * 400),
      });
      log("sync", "converged — <b>0</b> divergent ranges, cursor advanced", "ok");
      state.sync.queued = 0;
      return;
    }
    applySync({ progress: p, divergent: Math.max(0, Math.round((1 - p) * 9)) });
    setTimeout(step, 220);
  };
  step();
}

const SIM_EVENTS = [
  ["ingest", "chunked 3 new observations · sensitivity <b>internal</b>"],
  ["onnx", "embed batch(8) coalesced in 8 ms window · p95 <b>4.1 ms</b>"],
  ["policy", "1 point marked <b>restricted</b> — pinned local-only", "warn"],
  ["memory", "compactor moved 128 points HOT → WARM (int8)"],
  ["renewal", "freshness sweep · 3 points marked stale"],
  ["link", "connectivity oracle: probe timeout, holding <b>OFFLINE</b>"],
  ["sync", "op queued durably · idempotency key issued"],
  ["retrieval", "hybrid query served <b>locally</b> in 7 ms"],
  ["wal", "checkpoint written · 0 torn records", "ok"],
  ["thermal", "package 58°C · fp16 variant retained"],
];

function simulateLoop() {
  if (state.live) return;
  const [c, m, lvl] = SIM_EVENTS[Math.floor(Math.random() * SIM_EVENTS.length)];
  log(c, m, lvl);

  state.mem.hot += Math.floor(Math.random() * 9);
  state.mem.warm += Math.floor(Math.random() * 4);
  applyMemory({ ...state.mem, collections: 4 });

  state.sync.queued += Math.floor(Math.random() * 3);
  applySync({ queued: state.sync.queued, divergent: 4 + Math.floor(Math.random() * 5) });

  state.renewal.done = Math.min(state.renewal.total, state.renewal.done + Math.floor(Math.random() * 120));
  applyRenewal({
    done: state.renewal.done,
    total: state.renewal.total,
    progress: state.renewal.done / state.renewal.total,
    state: state.renewal.done >= state.renewal.total ? "COMPLETE" : "DUAL-SPACE",
  });
}

function seedSimulation() {
  state.mem = { hot: 18420, warm: 46110, cold: 122730 };
  applyMemory({ ...state.mem, collections: 4 });
  state.renewal = { state: "DUAL-SPACE", done: 41200, total: 187260, stale: 812, progress: 0.22 };
  applyRenewal({ ...state.renewal, progress: state.renewal.done / state.renewal.total });
  applySync({ state: "HOLDING", queued: 1284, divergent: 7, conflicts: 2, progress: 0, cursor: 84213 });
  applyNode({
    execution_provider: "XNNPACK (CPU)",
    embedder: "bge-small-en-v1.5",
    precision: "int8-dynamic",
    embed_p95_ms: 4.1,
    escalations: 0,
    node_id: "edge-07",
    reconnect_ms: 380,
  });
  log("boot", "node online · <b>local memory authoritative</b>", "ok");
  log("link", "no uplink — operating <b>offline-first</b>", "warn");
}

/* ── start ─────────────────────────────────────────────── */

function start() {
  seedSimulation();
  probe();
  setInterval(probe, 5000);
  setInterval(simulateLoop, 2600);
}

runBoot();
