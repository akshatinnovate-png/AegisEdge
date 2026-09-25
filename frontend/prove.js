/* ============================================================
   AegisEdge — PROVE IT
   Four claims, each with the button that would break it if it
   were false. A demo is an assertion; this is the part that
   invites the audience to check.
   ============================================================ */

const P = (id) => document.getElementById(id);
const pesc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const PAPI = () => (window.AEGIS_API || "http://localhost:8000").replace(/\/$/, "");

async function api(path, body, method) {
  const res = await fetch(PAPI() + "/api/v1" + path, {
    method: method || (body !== undefined ? "POST" : "GET"),
    headers: { "content-type": "application/json" },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  const text = await res.text();
  if (!res.ok) throw new Error(`${res.status} ${text}`.slice(0, 200));
  return text ? JSON.parse(text) : {};
}

const prove = {
  load: { rate: 0, timer: null, latencies: [], inflight: 0, sent: 0, failed: 0,
          skipped: 0, maxInflight: 96 },
  killing: false,
  timeline: null,
  // The verdict box has one owner at a time. The four-second poll used to
  // write to it as well, which meant the result of a kill — the thing the
  // whole panel exists to show — was overwritten a moment after it appeared.
  verdictOwner: null,
};

function verdict(owner, html, kind = "") {
  if (prove.verdictOwner && prove.verdictOwner !== owner) return false;
  P("durVerdict").className = `verdict${kind ? " verdict--" + kind : ""}`;
  P("durVerdict").innerHTML = html;
  return true;
}

/* ── 1. durability: the crowbar ──────────────────────────── */

const LADDER = ["FULL", "ECONOMISE", "TRIM", "ESSENTIAL", "SURVIVAL"];

async function refreshDurability() {
  try {
    const [stats, recovery] = await Promise.all([
      api("/memory/stats"), api("/integrity/recovery"),
    ]);
    P("durCount").textContent = (stats.total ?? 0).toLocaleString();
    P("durWal").textContent = (stats.wal?.lsn ?? 0).toLocaleString();
    P("durGen").textContent = recovery.generation ?? "—";
    P("durSupervised").innerHTML = recovery.supervised
      ? `<span class="badge badge--ok">supervised</span>`
      : `<span class="badge badge--hold">no supervisor — kill disabled</span>`;
    P("durKill").disabled = !recovery.supervised;
    if (recovery.recovered && recovery.generation > 1) {
      verdict("poll",
        `<b>${pesc(recovery.verdict)}</b><br><span style="color:var(--dim)">` +
        (recovery.deaths || []).slice(-1).map((d) =>
          `last death: ${pesc(d.signal_name || "exit " + d.exit_code)}` +
          `${d.hard_kill ? " (no handler ran, nothing flushed)" : ""}` +
          ` after ${d.lived_s}s alive`).join("") + `</span>`);
    }
  } catch (err) {
    verdict("poll", pesc(err.message), "bad");
  }
}

async function killNode() {
  if (prove.killing) return;
  prove.killing = true;
  prove.verdictOwner = "kill";          // the poll must not clobber this
  P("durKill").disabled = true;
  const before = P("durCount").textContent;
  verdict("kill", "sending SIGKILL…", "wait");
  try {
    const r = await api("/chaos/kill?sig=9&delay_ms=350", {});
    verdict("kill",
      `<b>SIGKILL in ${r.in_ms} ms.</b> ${r.resident_before.toLocaleString()} memories ` +
      `resident, WAL at ${r.wal_lsn_before}. No handler will run and nothing will be ` +
      `flushed. Waiting for the supervisor to start generation ${r.generation + 1}…`, "wait");
    await waitForNode(before);
  } catch (err) {
    verdict("kill", pesc(err.message), "bad");
    P("durKill").disabled = false;
  }
  prove.killing = false;
}

async function waitForNode(before) {
  const started = Date.now();
  for (let i = 0; i < 120; i += 1) {
    await new Promise((r) => setTimeout(r, 700));
    try {
      const recovery = await api("/integrity/recovery");
      const stats = await api("/memory/stats");
      const gap = ((Date.now() - started) / 1000).toFixed(1);
      const back = (stats.total ?? 0).toLocaleString();
      const intact = back === before;
      verdict("kill",
        `<b>Back in ${gap}s.</b> ${pesc(recovery.verdict)}<br>` +
        `<span style="color:${intact ? "var(--orange)" : "#e0785f"}">` +
        `${before} memories before the kill, ${back} after` +
        `${intact ? " — nothing was lost." : " — the difference is real and is not hidden."}` +
        `</span><div class="mono-note">A torn tail record, if there were one, is ` +
        `discarded rather than replayed: the WAL is fsynced <em>before</em> a write is ` +
        `acknowledged, so a half-written record is one whose caller never got an answer.</div>`,
        intact ? "" : "bad");
      P("durKill").disabled = false;
      return;
    } catch { /* still down — that is the point */ }
  }
  verdict("kill", "the node did not come back. Is it running under scripts/supervise.py?", "bad");
}

async function injectFault(name) {
  try {
    prove.verdictOwner = "fault";
    const r = await api(`/chaos/${name}`, { duration_s: 15 });
    verdict("fault", `<b>${pesc(name)}</b> injected · ` +
      `<span style="color:var(--dim)">${pesc(JSON.stringify(r).slice(0, 220))}</span>`);
  } catch (err) {
    verdict("fault", pesc(err.message), "bad");
  }
}

async function runFsck() {
  prove.verdictOwner = "fsck";
  verdict("fsck", "walking every segment…", "wait");
  try {
    const r = await api("/integrity/fsck", {});
    const clean = r.clean ?? r.ok ?? (r.corrupt === 0);
    verdict("fsck", `<b>fsck ${clean ? "clean" : "found damage"}</b> · ` +
      `<span style="color:var(--dim)">${pesc(JSON.stringify(r).slice(0, 300))}</span>`,
      clean ? "" : "bad");
  } catch (err) {
    verdict("fsck", pesc(err.message), "bad");
  }
}

/* ── 2. the load dial ────────────────────────────────────── */

const QUERIES = [
  "coolant pressure hard stop", "bearing vibration raceway", "what happened on line 2",
  "torque limit after the bearing job", "gantry re-homed interlock", "night shift readings",
  "spindle inrush current", "conveyor belt tension weekly",
];

function setRate(rate) {
  prove.load.rate = rate;
  P("dialValue").textContent = rate ? `${rate} q/s` : "idle";
  if (prove.load.timer) { clearInterval(prove.load.timer); prove.load.timer = null; }
  if (!rate) return;
  prove.load.skipped = 0;
  // One tick every 100 ms, issuing a tenth of the requested rate. Bursting a
  // whole second's worth at once would measure the burst, not the rate.
  const perTick = Math.max(1, Math.round(rate / 10));
  prove.load.timer = setInterval(() => {
    for (let i = 0; i < perTick; i += 1) fireQuery();
  }, 100);
}

async function fireQuery() {
  // A browser will happily queue thousands of fetches and then measure its own
  // queue. At 900 q/s this panel reported a p50 of 3.6 seconds with 1,333
  // requests outstanding while the node's own burn rate sat at zero — because
  // the node was fine and the *browser* was the bottleneck. Capping in-flight
  // requests measures the node instead, and what the cap turns away is counted
  // rather than quietly dropped.
  if (prove.load.inflight >= prove.load.maxInflight) { prove.load.skipped += 1; return; }
  const q = QUERIES[Math.floor(Math.random() * QUERIES.length)] + " " +
            (Math.random() < 0.5 ? "" : Math.floor(Math.random() * 1000));
  const t0 = performance.now();
  prove.load.inflight += 1;
  prove.load.sent += 1;
  try {
    await api("/search", { query: q, k: 5 });
    prove.load.latencies.push(performance.now() - t0);
    if (prove.load.latencies.length > 200) prove.load.latencies.shift();
  } catch { prove.load.failed += 1; }
  prove.load.inflight -= 1;
}

async function refreshLoad() {
  try {
    const slo = await api("/slo");
    const level = slo.level || "FULL";
    document.querySelectorAll("#ladder .rung").forEach((el) => {
      const idx = LADDER.indexOf(el.dataset.rung);
      el.dataset.on = el.dataset.rung === level;
      el.dataset.passed = idx < LADDER.indexOf(level);
    });
    P("sloReason").textContent = (slo.manual_override
      ? `pinned to ${slo.manual_override} — ` : "") + (slo.reason || "");
    P("sloShed").innerHTML = (slo.disabled_features || []).length
      ? (slo.disabled_features || []).map((f) => `<span class="badge badge--hold">${pesc(f)}</span>`).join(" ")
      : `<span class="badge badge--ok">every stage enabled</span>`;
    const lat = [...prove.load.latencies].sort((a, b) => a - b);
    const at = (q) => lat.length ? lat[Math.min(lat.length - 1, Math.floor(q * lat.length))] : 0;
    P("loadP50").textContent = at(0.5).toFixed(0);
    P("loadP99").textContent = at(0.99).toFixed(0);
    P("loadBurn").textContent = (slo.burn_rate ?? 0).toFixed(2);
    P("loadInflight").textContent = prove.load.skipped
      ? `${prove.load.inflight} (${prove.load.skipped.toLocaleString()} held back by the dial)`
      : prove.load.inflight;
    const target = slo.objective?.target_ms ?? 150;
    P("spark").innerHTML = lat.length
      ? [...prove.load.latencies].slice(-80).map((v) =>
          `<i style="height:${Math.min(100, (v / (target * 4)) * 100)}%" ` +
          `data-breach="${v > target}"></i>`).join("")
      : "";
  } catch { /* the node may be mid-restart; the next tick will find it */ }
}

/* ── 3. energy ───────────────────────────────────────────── */

async function refreshEnergy() {
  try {
    const e = await api("/energy");
    const q = e.by_kind?.query;
    P("energySource").innerHTML = e.measured
      ? `<span class="badge badge--ok">measured · ${pesc(e.source)}</span>`
      : `<span class="badge badge--hold">modelled</span>`;
    P("energyHow").textContent = e.how || "";
    if (q && q.per_battery_percent) {
      P("energyN").textContent = q.per_battery_percent.toLocaleString();
      P("energyU").textContent = "ANSWERS PER 1% OF BATTERY";
      P("energySub").textContent =
        `${q.millijoules_per_op} mJ each · ${q.cpu_ms_per_op} ms of CPU · ${q.ops.toLocaleString()} measured`;
    } else if (q) {
      P("energyN").textContent = q.millijoules_per_op;
      P("energyU").textContent = "mJ PER ANSWER";
      P("energySub").textContent =
        `${q.cpu_ms_per_op} ms of CPU · ${q.ops.toLocaleString()} answers measured · ` +
        `set AEGIS_BATTERY_CAPACITY_WH for the per-battery figure`;
    } else {
      P("energyN").textContent = "—";
      P("energySub").textContent = "no answers measured yet — run some load";
    }
  } catch { /* ignore */ }
}

/* ── 4. provenance ───────────────────────────────────────── */

async function refreshProvenance() {
  try {
    const p = await api("/provenance");
    P("provRoot").textContent = p.root;
    P("provFiles").textContent = p.source.files;
    P("provCommit").textContent = p.git.short || "—";
    P("provBranch").textContent = p.git.branch || "—";
    P("provClean").innerHTML = p.git.clean === true
      ? `<span class="badge badge--ok">clean</span>`
      : `<span class="badge badge--hold">${(p.git.dirty_files || []).length} uncommitted</span>`;
    P("provModels").textContent = p.models.artefacts;
  } catch (err) {
    P("provRoot").textContent = err.message;
  }
}

/* ── 5. time travel ──────────────────────────────────────── */

async function refreshTimeline() {
  const slider = P("ttSlider");
  const hoursBack = Number(slider.value);
  const when = Date.now() / 1000 - hoursBack * 3600;
  P("ttWhen").textContent = hoursBack === 0
    ? "now" : `${hoursBack}h ago · ${new Date(when * 1000).toLocaleString()}`;
  try {
    const [asOf, diff] = await Promise.all([
      api(`/graph/as-of?when=${when}`),
      hoursBack > 0 ? api(`/graph/diff?earlier=${when}`) : Promise.resolve(null),
    ]);
    P("ttFacts").textContent = (asOf.facts ?? asOf.live_facts ?? 0).toLocaleString();
    P("ttEntities").textContent = (asOf.entities ?? 0).toLocaleString();
    P("ttDiff").innerHTML = diff
      ? `<div class="kv"><span>learned since</span><b>${(diff.learned || []).length}</b>` +
        `<span>retracted since</span><b>${(diff.retracted || []).length}</b></div>` +
        (diff.learned || []).slice(0, 4).map((f) =>
          `<div class="mono-note">+ ${pesc(f.subject)} <em>${pesc(f.predicate)}</em> ${pesc(f.object)}</div>`).join("")
      : `<div class="mono-note">Drag back to compare what the device believed then with what it believes now.</div>`;
  } catch (err) {
    P("ttDiff").innerHTML = `<div class="mono-note">${pesc(err.message)}</div>`;
  }
}

/* ── wiring ─────────────────────────────────────────────── */


/* ── 6. a peer that lies ─────────────────────────────────── */

const ATTACK_LABEL = {
  honest: "an honest operation",
  tamper: "an operation rewritten in flight",
  impersonate: "an operation written in another device's name",
  unsigned: "an unsigned operation",
};

async function refreshMeshIdentity() {
  try {
    const mesh = await api("/mesh/status");
    const id = mesh.identity || {};
    P("atkDevice").textContent = id.device_id || "—";
    P("atkKeys").textContent = (id.known_devices || []).length || "—";
    P("atkVerified").textContent = mesh.verified_ops ?? "—";
    P("atkForged").textContent = mesh.refused_forged ?? "—";
    P("atkPolicy").textContent = mesh.refused_inbound ?? "—";
  } catch {
    ["atkDevice", "atkKeys", "atkVerified", "atkForged", "atkPolicy"]
      .forEach((k) => { P(k).textContent = "—"; });
  }
}

async function mountAttack(kind) {
  const box = P("atkVerdict");
  box.className = "verdict verdict--wait";
  box.innerHTML = `sending ${ATTACK_LABEL[kind]}…`;
  try {
    const r = await api("/mesh/attack", { kind });
    await refreshMeshIdentity();
    // "Correct" is the node behaving as claimed, which for the honest case
    // means accepting. Reporting "refused = good" would make a node that
    // refuses everything look perfect.
    const good = r.correct;
    box.className = `verdict verdict--${good ? "ok" : "bad"}`;
    box.innerHTML = good
      ? (r.accepted
        ? `<b>Accepted</b>, and it should be — ${r.what_it_did}. This is the control: ` +
          `the refusals below only mean something next to an acceptance here.`
        : `<b>Refused.</b> ${r.what_it_did[0].toUpperCase()}${r.what_it_did.slice(1)}. ` +
          `The relay cannot produce the author's signature over content the author ` +
          `never wrote, so the node has something to check rather than somebody's word ` +
          `to take. Counted as a forgery, not dropped quietly.`)
      : `<b>Wrong outcome.</b> Expected this to be ${r.expected} and it was ` +
        `${r.accepted ? "accepted" : "refused"}. That is a real defect, and it is on ` +
        `screen rather than in a log.`;
  } catch (err) {
    box.className = "verdict verdict--bad";
    box.textContent = err.message;
  }
}

function mountProve() {
  P("durKill").addEventListener("click", killNode);
  P("durFsck").addEventListener("click", runFsck);
  document.querySelectorAll("[data-fault]").forEach((b) =>
    b.addEventListener("click", () => injectFault(b.dataset.fault)));
  P("dial").addEventListener("input", (e) => setRate(Number(e.target.value)));
  document.querySelectorAll("[data-pin]").forEach((b) =>
    b.addEventListener("click", async () => {
      // An explicit pin, and the panel says so. Showing what a rung disables is
      // useful; implying that load caused it would not be.
      try {
        await api("/slo/override", { level: b.dataset.pin || null });
        refreshLoad();
      } catch (err) { P("sloReason").textContent = err.message; }
    }));
  P("ttSlider").addEventListener("input", refreshTimeline);
  document.querySelectorAll("[data-attack]").forEach((b) =>
    b.addEventListener("click", () => mountAttack(b.dataset.attack)));

  setInterval(() => {
    if (document.body.dataset.mode !== "prove") return;
    refreshLoad(); refreshEnergy();
  }, 1000);
  setInterval(() => {
    if (document.body.dataset.mode !== "prove") return;
    refreshDurability(); refreshMeshIdentity();
  }, 4000);
}

document.addEventListener("DOMContentLoaded", () => {
  mountProve();
  document.querySelectorAll('.mode-switch button[data-mode="prove"]').forEach((b) =>
    b.addEventListener("click", () => {
      refreshDurability(); refreshEnergy(); refreshProvenance();
      refreshLoad(); refreshTimeline(); refreshMeshIdentity();
    }));
});
