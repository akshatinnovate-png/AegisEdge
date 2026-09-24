/* ============================================================
   AegisEdge — USE mode
   The console answers "is it working". This answers "what is it
   for", and shows the machinery beside the action that caused it.

   Two devices are two real node processes, not two tabs against
   one. They find each other over the mesh and reconcile directly,
   with no cloud in the middle — which is the part of the problem
   statement that is easiest to claim and hardest to demonstrate.
   ============================================================ */

const DEVICES = (window.AEGIS_DEVICES || [
  { key: "A", label: "DEVICE A", api: "http://localhost:8201" },
  { key: "B", label: "DEVICE B", api: "http://localhost:8202" },
]);

const use = {
  active: DEVICES[0].key,
  nodes: {},                 // key -> last /health + /mesh/status
  rail: [],                  // most recent first
};

const U = (id) => document.getElementById(id);
const dev = (key) => DEVICES.find((d) => d.key === key);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

async function call(key, path, body, method) {
  const d = dev(key);
  const res = await fetch(d.api + "/api/v1" + path, {
    method: method || (body !== undefined ? "POST" : "GET"),
    headers: { "content-type": "application/json" },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`.slice(0, 160));
  return res.json();
}

/* ── the rail: every action, and what the node did about it ── */

function pushStep(what, rows, note) {
  use.rail.unshift({ what, rows, note, at: new Date() });
  use.rail = use.rail.slice(0, 24);
  renderRail();
}

function renderRail() {
  const body = U("traceBody");
  if (!use.rail.length) {
    body.innerHTML = `<div class="trace__empty">Nothing yet. Save a memory or ask a
      question — every field the node set, and every decision it made, appears here
      beside the action that caused it.</div>`;
    return;
  }
  body.innerHTML = use.rail.map((s) => `
    <div class="step">
      <div class="step__what">${esc(s.what)} · ${s.at.toLocaleTimeString()}</div>
      ${s.rows.map(([k, v]) => `<div class="step__row"><span>${esc(k)}</span><b>${v}</b></div>`).join("")}
      ${s.note ? `<div class="step__note">${s.note}</div>` : ""}
    </div>`).join("");
}

function stageBars(stages) {
  if (!stages) return "";
  const entries = Object.entries(stages).filter(([, v]) => typeof v === "number");
  const max = Math.max(...entries.map(([, v]) => v), 0.001);
  return entries.map(([k, v]) => `
    <div class="stage">
      <span>${esc(k.replace(/_ms$/, ""))}</span>
      <span class="stage__bar"><span class="stage__fill" style="width:${(v / max) * 100}%"></span></span>
      <span class="stage__ms">${v.toFixed(2)}</span>
    </div>`).join("");
}

/* ── devices ────────────────────────────────────────────── */

async function refreshDevice(key) {
  const card = U("dev" + key);
  try {
    const [health, mesh] = await Promise.all([call(key, "/health"), call(key, "/mesh/status")]);
    use.nodes[key] = { health, mesh, reachable: true };
    const offline = !!(mesh.link && mesh.link.offline);
    card.querySelector("[data-f=id]").textContent = health.node_id || "—";
    card.querySelector("[data-f=points]").textContent = health.points ?? "—";
    card.querySelector("[data-f=peers]").innerHTML = mesh.alive_peers
      ? `<span class="badge badge--ok">${mesh.alive_peers} reachable</span>`
      : `<span class="badge badge--hold">no peer</span>`;
    card.querySelector("[data-f=ops]").textContent = mesh.known_ops ?? "—";
    const radio = card.querySelector("input[type=checkbox]");
    radio.checked = !offline;
    card.querySelector(".radio__label").textContent = offline ? "RADIO OFF" : "RADIO ON";
  } catch (err) {
    use.nodes[key] = { reachable: false };
    card.querySelector("[data-f=id]").textContent = "unreachable";
    card.querySelector("[data-f=peers]").innerHTML =
      `<span class="badge badge--hold">node down</span>`;
  }
}

async function toggleRadio(key, on) {
  try {
    const r = await call(key, `/mesh/offline?on=${on ? "false" : "true"}`, {});
    pushStep(on ? "RADIO ON" : "RADIO PULLED", [
      ["device", esc(dev(key).label)],
      ["mesh offline", String(r.offline)],
    ], r.offline
      ? "Peers are unreachable. Local memory is still authoritative: capture and retrieval carry on, and operations queue for the next reconciliation."
      : "Peers reachable again. Nothing was lost while it was down.");
  } catch (err) {
    pushStep("RADIO", [["failed", esc(err.message)]]);
  }
  await refreshAll();
}

async function introduce() {
  // Tell each device where the other one lives. This is the whole of
  // discovery: a real deployment gets these from mDNS or a fleet roster.
  const [a, b] = DEVICES;
  try {
    const [ha, hb] = await Promise.all([call(a.key, "/health"), call(b.key, "/health")]);
    await call(a.key, "/mesh/peers", { node_id: hb.node_id, endpoint: b.api });
    await call(b.key, "/mesh/peers", { node_id: ha.node_id, endpoint: a.api });
    pushStep("PEERED", [["A", esc(ha.node_id)], ["B", esc(hb.node_id)]],
      "Each device now knows where the other lives. They gossip directly — no cloud, no coordinator.");
  } catch (err) {
    pushStep("PEERING FAILED", [["error", esc(err.message)]],
      "Both devices must be running. See the two-device command in the README.");
  }
  await refreshAll();
}

async function reconcile() {
  const rounds = [];
  for (const d of DEVICES) {
    try {
      const r = await call(d.key, "/mesh/round?peers=2", {});
      (r.rounds || []).forEach((x) => rounds.push([d.key, x]));
    } catch (err) { rounds.push([d.key, { error: err.message.slice(0, 80) }]); }
  }
  const rows = rounds.map(([k, r]) => [
    `${k} → ${esc(r.peer || "?")}`,
    r.error ? `<span class="badge badge--hold">${esc(r.error)}</span>`
            : `↓${r.pulled ?? 0} ↑${r.pushed ?? 0}`,
  ]);
  pushStep("RECONCILED", rows,
    "Anti-entropy over IBLT digests: each side learns only the operations it is missing, not the whole log.");
  await refreshAll();
}

const refreshAll = () => Promise.all(DEVICES.map((d) => refreshDevice(d.key)));

/* ── capture ────────────────────────────────────────────── */

async function remember() {
  const text = U("useText").value.trim();
  if (!text) return;
  const collection = U("useCollection").value;
  U("useSave").disabled = true;
  try {
    const t0 = performance.now();
    const r = await call(use.active, "/memory/ingest", { text, collection });
    const ms = performance.now() - t0;
    const held = r.sync_class !== "sync_full";
    pushStep("REMEMBERED", [
      ["id", esc(r.id)],
      ["collection", esc(r.collection)],
      ["tier", esc(r.tier)],
      ["sensitivity", esc(r.sensitivity)],
      ["handling", `<span class="badge ${held ? "badge--hold" : "badge--ok"}">${esc(r.sync_class)}</span>`],
      ["policy rule", esc(r.policy_rule)],
      ["took", `${ms.toFixed(0)} ms`],
    ], held
      ? `The classifier read this as <em>${esc(r.sensitivity)}</em>, so policy rule
         <em>${esc(r.policy_rule)}</em> holds it as <em>${esc(r.sync_class)}</em> — it is
         searchable here and will not leave this device in full.`
      : `Written to the WAL before it became searchable, embedded locally, and queued
         for the next reconciliation.`);
    U("useText").value = "";
  } catch (err) {
    pushStep("WRITE REFUSED", [["error", esc(err.message)]]);
  }
  U("useSave").disabled = false;
  await refreshAll();
}

/* ── ask ────────────────────────────────────────────────── */

async function ask() {
  const query = U("useAsk").value.trim();
  if (!query) return;
  U("useAskGo").disabled = true;
  U("useAnswer").innerHTML = `<div class="trace__empty">thinking on-device…</div>`;
  try {
    const r = await call(use.active, "/ask", { query });
    const abstained = !r.answer || r.confidence < 0.35;
    U("useAnswer").innerHTML = `
      <div class="answer ${abstained ? "answer--abstain" : ""}">
        <div class="answer__text">${abstained
          ? "I don't have enough in local memory to answer that confidently."
          : esc(r.answer)}</div>
        <div class="answer__meta">
          <span>confidence ${(r.confidence * 100).toFixed(0)}%</span>
          <span>${r.latency_ms.toFixed(1)} ms</span>
          <span>${r.escalated ? "escalated to cloud" : "answered locally"}</span>
          ${r.contradictions?.length ? `<span>${r.contradictions.length} contradiction(s)</span>` : ""}
        </div>
      </div>
      ${(r.citations || []).map((c) => `
        <div class="cite">
          <div class="cite__head"><span>${esc(c.collection)} · ${esc(c.id)}</span>
            <span>${c.score.toFixed(3)} · ${(c.matched_by || []).join("+")}</span></div>
        </div>`).join("")}`;
    pushStep("ASKED", [
      ["confidence", `${(r.confidence * 100).toFixed(0)}%`],
      ["citations", String((r.citations || []).length)],
      ["latency", `${r.latency_ms.toFixed(1)} ms`],
      ["where", r.escalated ? "cloud" : "on-device"],
    ], abstained
      ? "Answer withheld. Conformal calibration puts this below the coverage threshold, and a confident wrong answer is worse than an admission."
      : "Retrieved, reranked, and answered from cited memories. Every claim points at the memory it came from.");
  } catch (err) {
    U("useAnswer").innerHTML = `<div class="trace__empty">${esc(err.message)}</div>`;
  }
  U("useAskGo").disabled = false;
}

/* ── recall ─────────────────────────────────────────────── */

async function recall() {
  const query = U("useFind").value.trim();
  if (!query) return;
  try {
    const r = await call(use.active, "/search", { query, k: 6 });
    U("useHits").innerHTML = (r.results || []).map((h) => `
      <div class="hit">
        <div class="hit__head"><span>${esc(h.collection)} · ${esc(h.tier)} · ${esc(h.id)}</span>
          <span>${h.score.toFixed(3)}</span></div>
        <div class="hit__text">${esc(h.text)}</div>
      </div>`).join("") || `<div class="trace__empty">no memories matched</div>`;
    const u = r.understanding || {};
    const repaired = u.corrections && Object.keys(u.corrections).length;
    const relaxed = u.inference_relaxed;
    pushStep("RECALLED", [
      ["hits", String((r.results || []).length)],
      ["latency", `${r.latency_ms.toFixed(1)} ms`],
      ["mode", esc(r.mode)],
      ["cached", String(!!r.cached)],
    ], `${stageBars(r.stages)}
        ${repaired ? `<div style="margin-top:8px">Spelling repaired against the local
           vocabulary: <em>${esc(JSON.stringify(u.corrections))}</em> — no network, no
           spell-check service.</div>` : ""}
        ${relaxed ? `<div style="margin-top:8px">An inferred narrowing matched nothing and
           was dropped: <em>${esc(JSON.stringify(relaxed.dropped))}</em>. A guess may not
           silently replace results with an empty page.</div>` : ""}`);
  } catch (err) {
    U("useHits").innerHTML = `<div class="trace__empty">${esc(err.message)}</div>`;
  }
}

/* ── wiring ─────────────────────────────────────────────── */

function mountUse() {
  U("devices").innerHTML = DEVICES.map((d) => `
    <div class="device" id="dev${d.key}" data-active="${d.key === use.active}">
      <div class="device__top">
        <div><div class="device__name">${esc(d.label)}</div>
          <div class="device__id" data-f="id">—</div></div>
        <label class="radio">
          <input type="checkbox" data-radio="${d.key}" checked />
          <span class="radio__track"></span><span class="radio__label">RADIO ON</span>
        </label>
      </div>
      <div class="device__row"><span>memories</span><b data-f="points">—</b></div>
      <div class="device__row"><span>known ops</span><b data-f="ops">—</b></div>
      <div class="device__row"><span>peers</span><b data-f="peers">—</b></div>
      <div class="device__actions">
        <button data-use="${d.key}">USE THIS ONE</button>
      </div>
    </div>`).join("");

  document.querySelectorAll("[data-radio]").forEach((el) =>
    el.addEventListener("change", (e) => toggleRadio(e.target.dataset.radio, e.target.checked)));
  document.querySelectorAll("[data-use]").forEach((el) =>
    el.addEventListener("click", (e) => {
      use.active = e.target.dataset.use;
      document.querySelectorAll(".device").forEach((c) =>
        c.setAttribute("data-active", c.id === "dev" + use.active));
      U("activeDevice").textContent = dev(use.active).label;
    }));

  U("usePeer").addEventListener("click", introduce);
  U("useReconcile").addEventListener("click", reconcile);
  U("useSave").addEventListener("click", remember);
  U("useAskGo").addEventListener("click", ask);
  U("useFindGo").addEventListener("click", recall);
  U("useAsk").addEventListener("keydown", (e) => e.key === "Enter" && ask());
  U("useFind").addEventListener("keydown", (e) => e.key === "Enter" && recall());

  document.querySelectorAll(".tabs button").forEach((b) =>
    b.addEventListener("click", () => {
      document.querySelectorAll(".tabs button").forEach((x) =>
        x.setAttribute("aria-selected", x === b));
      document.querySelectorAll(".pane").forEach((p) =>
        p.setAttribute("data-open", p.id === "pane-" + b.dataset.pane));
    }));

  U("activeDevice").textContent = dev(use.active).label;
  renderRail();
  refreshAll();
  setInterval(() => { if (document.body.dataset.mode === "use") refreshAll(); }, 4000);
}

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".mode-switch button").forEach((b) =>
    b.addEventListener("click", () => {
      document.body.dataset.mode = b.dataset.mode;
      document.querySelectorAll(".mode-switch button").forEach((x) =>
        x.setAttribute("aria-pressed", x === b));
      if (b.dataset.mode === "use") refreshAll();
    }));
  mountUse();
});
