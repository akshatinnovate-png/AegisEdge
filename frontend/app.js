/* ============================================================
   AegisEdge — console frontend
   Every number on this page comes from the node. When the node is
   unreachable the console says so and shows nothing: a dashboard
   that invents plausible values is worse than a blank one, because
   you cannot tell the difference until it matters.
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

// The boot screen reads the node's own /health while it plays, so each
// line is filled in from the real answer rather than asserted.
const POST_LINES = [
  ["AegisEdge Node Console", "hi"],
  ["Code Cubicle 6.0 — PS03", ""],
  ["", ""],
  ["Contacting node........................ ", "probe", "node_id"],
  ["Execution provider..................... ", "probe", "execution_provider"],
  ["Embedding model........................ ", "probe", "model"],
  ["Vector store........................... ", "probe", "memory_backend"],
  ["Resident memories...................... ", "probe", "points"],
  ["Uplink................................. ", "probe", "link"],
  ["Degradation level...................... ", "probe", "degradation"],
  ["", ""],
  ["Local memory is authoritative. Network optional.", "hi"],
];

// Filled by the first /health call; the boot screen renders whatever is here.
let BOOT_FACTS = null;

function bootValue(key) {
  if (!BOOT_FACTS) return "…";
  if (key === "model") {
    const m = BOOT_FACTS.model || {};
    return m.source ? `${m.source} · ${m.dim}d` : "unknown";
  }
  const value = BOOT_FACTS[key];
  return value === undefined || value === null ? "—" : String(value);
}

const CAPTIONS = [
  "contacting node…",
  "reading model provenance…",
  "reading vector store…",
  "reading uplink state…",
  "haze clearing…",
];

async function runBoot() {
  const post = $("post");
  const caption = $("bootCaption");
  let i = 0;
  try {
    BOOT_FACTS = await get("/api/v1/health", 4000);
  } catch {
    BOOT_FACTS = null;                      // the boot screen will say so
  }
  paintBootFooter();

  const tick = () => {
    if (i < POST_LINES.length) {
      const [text, cls, tail] = POST_LINES[i];
      const row = document.createElement("div");
      row.textContent = text;
      if (cls === "hi") row.className = "hi";
      if (tail) {
        const s = document.createElement("span");
        const value = cls === "probe" ? bootValue(tail) : tail;
        s.className = BOOT_FACTS || cls !== "probe" ? "ok" : "warn";
        s.textContent = "[ " + (BOOT_FACTS || cls !== "probe" ? value : "NO NODE") + " ]";
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

function paintBootFooter() {
  const provider = $("bootProvider");
  if (!provider) return;
  provider.textContent = BOOT_FACTS
    ? `ONNX RUNTIME · ${BOOT_FACTS.execution_provider || "unknown"}`
    : "NODE UNREACHABLE";
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
    if (!state.live) log("node", `node reachable in ${rtt} ms — reading live state`, "ok");
    state.live = true;
    state.retries = 0;
    // The pill tracks the NODE's uplink, not the browser's hop to it: a node
    // we can reach may still be offline from the cloud, which is the normal
    // case for this product. /node/state supplies that below.
    $("dataSrc").textContent = "SOURCE · LIVE NODE · " + (h.memory_backend || "");
    if (h.execution_provider) $("epName").textContent = h.execution_provider;
    if (h.node_id) $("nodeId").textContent = h.node_id;
    await pullAll();
    openSocket();
  } catch {
    if (state.live) log("node", "node unreachable — console is showing nothing", "warn");
    state.live = false;
    setLink("offline", null);
    $("dataSrc").textContent = "NO NODE · " + API;
    blankOut();
    state.retries++;
  }
}

function blankOut() {
  // Clear every figure rather than leave the last known value on screen
  // looking current. A stale number is indistinguishable from a live one.
  for (const id of ["memTotal", "tHot", "tWarm", "tCold", "syncQueued", "syncDiv",
                    "syncConf", "renStale", "escCount"]) {
    const el = $(id);
    if (el) el.textContent = "—";
  }
  for (const id of ["epName", "embName", "precName", "embP95", "syncState", "renState", "renDone"]) {
    const el = $(id);
    if (el) el.textContent = "—";
  }
  for (const id of ["syncBar", "renBar"]) {
    const el = $(id);
    if (el) el.style.width = "0%";
  }
  for (const id of ["tHot", "tWarm", "tCold"]) {
    const bar = $(id) && $(id).parentElement.querySelector("i");
    if (bar) bar.style.width = "0%";
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
      if (msg.sync) applySync(msg.sync);
      if (msg.kind === "heartbeat") return;
      if (msg.message) log("node", msg.message, msg.level);
      break;
    case "sync":
      applySync(msg);
      if (msg.message) log("sync", msg.message, msg.level);
      break;
    case "renewal":
      applyRenewal(msg);
      if (msg.message) log("renewal", msg.message, msg.level);
      break;
    case "link":
      if (msg.state) setLink(msg.state, msg.rtt_ms);
      if (msg.reconnect_ms != null) $("reconnMs").textContent = Math.round(msg.reconnect_ms);
      if (msg.message) log("link", msg.message, msg.level);
      break;
    case "memory":
    case "search":
    case "reasoning":
    case "alerts":
    default:
      if (msg.message) log(msg.channel || msg.kind || "node", msg.message, msg.level);
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
  if (n.link && n.link.state) setLink(n.link.state, n.link.rtt_ms);
  if (n.execution_provider) $("epName").textContent = n.execution_provider;
  if (n.embedder) $("embName").textContent = n.embedder;
  if (n.governor && n.governor.token_budget)
    $("precName").textContent = `${n.governor.token_budget} tok · ${n.governor.rung}`;
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
  if (s.link) setLink(s.link, null);
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
  try {
    payload = await post("/api/v1/search", { query: q, k: 5, mode: "hybrid" });
  } catch (err) {
    $("searchMeta").textContent = "node did not answer — no results to show";
    log("search", `query failed: ${err}`, "warn");
    return;
  }

  const ms = payload.latency_ms != null ? payload.latency_ms : Math.round(performance.now() - t0);
  $("searchMeta").textContent =
    `${payload.results.length} hits · ${ms} ms · dense+sparse · RRF · rerank` +
    (payload.escalated
      ? " · ESCALATED → TRITON"
      : " · LOCAL" + (payload.escalation && payload.escalation.reason ? " (" + payload.escalation.reason + ")" : ""));

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
  try {
    applySync(await post("/api/v1/sync/trigger", {}));
  } catch {
    log("sync", "node unreachable — ops stay queued locally", "warn");
  }
  setTimeout(() => (btn.disabled = false), 2600);
});

/* ── start ─────────────────────────────────────────────── */

function start() {
  probe();
  setInterval(probe, 5000);
}

runBoot();
