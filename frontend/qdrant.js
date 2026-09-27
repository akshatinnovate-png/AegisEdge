/* ============================================================
   AegisEdge — the Qdrant engine panel

   Everything drawn here comes from /api/v1/qdrant and from the plan the
   engine returned for the last query. Two rules the panel keeps:

   A stage is drawn inside the engine boundary only when the plan says it
   ran there. When the query took the local index instead, the boundary
   goes dim and the stages move outside it — the graphic is a record of
   where the work happened, not an illustration of the architecture.

   Nothing is invented while the node is unreachable. The plan clears and
   the caption says so, because a dashboard that keeps drawing the last
   good answer is indistinguishable from one that is working.
   ============================================================ */

(() => {
  const el = (id) => document.getElementById(id);
  const panel = el("qdPanel");
  if (!panel) return;

  const NS = "http://www.w3.org/2000/svg";
  const W = 520, H = 190;

  // node layout, in the plan's own reading order
  const LAYOUT = {
    query:  { x: 6,   y: 78,  w: 72,  h: 36, title: "QUERY",    detail: "text" },
    dense:  { x: 116, y: 22,  w: 132, h: 40, title: "PREFETCH", detail: "dense · hnsw" },
    lex:    { x: 116, y: 128, w: 132, h: 40, title: "PREFETCH", detail: "lex · postings" },
    fuse:   { x: 274, y: 75,  w: 96,  h: 42, title: "FUSE",     detail: "rrf" },
    late:   { x: 388, y: 75,  w: 122, h: 42, title: "MAXSIM",   detail: "late · maxsim" },
  };
  const EDGES = [
    ["query", "dense"], ["query", "lex"],
    ["dense", "fuse"], ["lex", "fuse"], ["fuse", "late"],
  ];

  const state = { plan: null, live: false, path: "", engine: "", refusals: 0 };

  function centre(key, side) {
    const n = LAYOUT[key];
    return { x: side === "out" ? n.x + n.w : n.x, y: n.y + n.h / 2 };
  }

  function draw() {
    const svg = el("qdPlan");
    svg.textContent = "";
    const plan = state.plan;
    const inEngine = !!(plan && plan.stages && plan.stages.length && !plan.fell_back);
    const stageBy = {};
    (plan && plan.stages ? plan.stages : []).forEach((s) => {
      if (s.name.startsWith("dense")) stageBy.dense = s;
      else if (s.name.startsWith("sparse")) stageBy.lex = s;
      else if (s.name === "fusion") stageBy.fuse = s;
      else if (s.name.startsWith("late")) stageBy.late = s;
    });

    // engine boundary: it encloses the stages only while they ran inside it
    const box = document.createElementNS(NS, "rect");
    box.setAttribute("x", 104); box.setAttribute("y", 8);
    box.setAttribute("width", W - 106); box.setAttribute("height", H - 22);
    box.setAttribute("rx", 3);
    box.setAttribute("class", "qd-engine" + (inEngine ? "" : " qd-engine--idle"));
    svg.appendChild(box);

    const label = document.createElementNS(NS, "text");
    label.setAttribute("x", 112); label.setAttribute("y", 21);
    label.setAttribute("class", "qd-engine-label");
    label.textContent = inEngine
      ? (state.engine || "QDRANT").toUpperCase() +
        ` · ${plan.queries_in_call || 1} QUERIES, ONE CALL` +
        (plan.total_ms != null ? ` · ${plan.total_ms.toFixed(2)}ms FOR THE CALL` : "")
      : "QDRANT · NOT ON THIS QUERY";
    svg.appendChild(label);

    EDGES.forEach(([from, to]) => {
      const a = centre(from, "out"), b = centre(to, "in");
      const mx = (a.x + b.x) / 2;
      const d = `M${a.x} ${a.y} C${mx} ${a.y} ${mx} ${b.y} ${b.x} ${b.y}`;
      const live = inEngine && (to === "late" ? !!stageBy.late : !!stageBy[to] || to === "fuse");
      const path = document.createElementNS(NS, "path");
      path.setAttribute("d", d);
      path.setAttribute("class", "qd-edge" + (live ? " on" : ""));
      svg.appendChild(path);
      if (live) {
        const flow = document.createElementNS(NS, "path");
        flow.setAttribute("d", d);
        flow.setAttribute("class", "qd-flow");
        svg.appendChild(flow);
      }
    });

    Object.entries(LAYOUT).forEach(([key, n]) => {
      const stage = stageBy[key];
      const on = key === "query" ? true : inEngine && !!stage;
      const g = document.createElementNS(NS, "g");
      g.setAttribute("class", "qd-node " + (on ? "on" : "off"));

      const rect = document.createElementNS(NS, "rect");
      rect.setAttribute("x", n.x); rect.setAttribute("y", n.y);
      rect.setAttribute("width", n.w); rect.setAttribute("height", n.h);
      rect.setAttribute("rx", 2);
      g.appendChild(rect);

      const t = document.createElementNS(NS, "text");
      t.setAttribute("x", n.x + 8); t.setAttribute("y", n.y + 15);
      t.setAttribute("class", "t");
      t.textContent = n.title;
      g.appendChild(t);

      const d = document.createElementNS(NS, "text");
      d.setAttribute("x", n.x + 8); d.setAttribute("y", n.y + 27);
      d.setAttribute("class", "d");
      d.textContent = stage && stage.limit ? `${n.detail} · limit ${stage.limit}` : n.detail;
      g.appendChild(d);

      // No per-stage millisecond here, ever. The engine runs all of this in one
      // call and returns one timing for it, so a number on each box would be a
      // total divided by four wearing a measurement's clothes. The call's real
      // timing is on the engine boundary above.
      svg.appendChild(g);
    });
  }

  function caption() {
    const c = el("qdCaption");
    if (!state.live) {
      c.innerHTML = "node unreachable — <b>no plan to show</b>";
      return;
    }
    const plan = state.plan;
    if (!plan) {
      c.innerHTML = "run a retrieval above and this draws the plan it executed";
      return;
    }
    if (plan.fell_back) {
      c.innerHTML =
        `the engine <b>declined</b> this query — ${escapeHtml(plan.fell_back)} — so it ran on the ` +
        `local index. A declined query is never answered with fewer results, only on the other path.`;
      return;
    }
    const total = plan.total_ms != null ? plan.total_ms.toFixed(2) : "—";
    // `calls` is one per collection searched: four stages each, not four stages
    // times four. Saying "one call" while the plan says four would be the kind
    // of rounding this console exists to avoid.
    const calls = plan.calls === 1
      ? "<b>1 call</b>"
      : `<b>${plan.calls} calls</b>, one per collection searched`;
    c.innerHTML =
      `${calls} · ${plan.stages.length} stages in the engine · ` +
      `<b>${total} ms</b> · ${plan.queries_in_call} queries batched per call · ` +
      `fusion ${escapeHtml(plan.fusion)} · rerank ${escapeHtml(plan.rerank)}` +
      (plan.prefilter ? " · payload pre-filter applied engine-side" : "");
  }

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = String(text == null ? "" : text);
    return div.innerHTML;
  }

  function badges(data) {
    const wrap = el("qdBadges");
    wrap.textContent = "";
    const add = (text, kind) => {
      const b = document.createElement("span");
      b.className = "qd__badge" + (kind ? " qd__badge--" + kind : "");
      b.textContent = text;
      wrap.appendChild(b);
    };
    add((data.backend || "qdrant").toUpperCase());
    const vectors = (data.hybrid && data.hybrid.vectors) || [];
    vectors.forEach((v) => add(v.toUpperCase(), "idx"));
    const idx = data.payload_indexes || {};
    if (idx.ignored_by_local_mode) {
      // The client says payload indexes do nothing in local mode. Saying so is
      // the difference between reporting an optimisation and claiming one.
      add(`${idx.ignored_by_local_mode} PAYLOAD IDX INERT (LOCAL MODE)`, "warn");
    } else if (idx.live) {
      add(`${idx.live} PAYLOAD IDX LIVE`, "idx");
    }
    const legacy = Object.values(data.collections || {})
      .filter((c) => c.schema && !String(c.schema).startsWith("hybrid")).length;
    if (legacy) add(`${legacy} LEGACY SCHEMA`, "warn");
  }

  function facets(rows) {
    const wrap = el("qdFacets");
    const head = el("qdFacetHead");
    wrap.textContent = "";
    head.hidden = !(rows && rows.length);
    if (!rows || !rows.length) return;
    const peak = Math.max(...rows.map((r) => r.count)) || 1;
    rows.forEach((row) => {
      const line = document.createElement("div");
      line.className = "qd-facet";
      line.innerHTML = "<span></span><i></i><b></b>";
      line.querySelector("span").textContent = row.value;
      line.querySelector("b").textContent = row.count;
      wrap.appendChild(line);
      requestAnimationFrame(() => {
        line.querySelector("i").style.width = Math.round((row.count / peak) * 100) + "%";
      });
    });
  }

  async function refresh() {
    let data;
    try {
      data = await get("/api/v1/qdrant", 2500);
    } catch {
      state.live = false;
      state.plan = null;
      el("qdBadges").textContent = "";
      el("qdPath").textContent = "—";
      el("qdPoints").textContent = "—";
      el("qdVectors").textContent = "—";
      el("qdLate").textContent = "—";
      draw();
      caption();
      return;
    }
    state.live = true;
    state.engine = data.backend;
    state.plan = data.last_plan;
    const qp = data.query_path || {};
    state.path = qp.policy || "";
    state.refusals = qp.engine_refusals || 0;
    badges(data);
    el("qdPath").textContent =
      `${qp.policy} · engine ${qp.engine_queries} · index ${qp.interpreter_queries}` +
      (qp.engine_refusals ? ` · declined ${qp.engine_refusals}` : "");
    const points = Object.values(data.collections || {})
      .reduce((sum, c) => sum + (c.points || 0), 0);
    el("qdPoints").textContent = points.toLocaleString();
    el("qdVectors").textContent = ((data.hybrid && data.hybrid.vectors) || []).join(" + ");
    const late = (data.hybrid && data.hybrid.late_tokens) || 0;
    el("qdLate").textContent = late
      ? `${late} tokens · ${(data.hybrid.late_bytes_per_point / 1024).toFixed(1)} KB/memory`
      : "off — a full residual costs more than the memory does";
    draw();
    caption();
    try {
      const f = await get("/api/v1/qdrant/facets?collection=episodic&key=sensitivity", 2500);
      facets(f.hits);
    } catch { /* facets are an extra; their absence is not a failure to report */ }
  }

  async function race(query) {
    const btn = el("qdRaceBtn");
    const verdict = el("qdVerdict");
    btn.disabled = true;
    verdict.className = "qd__verdict";
    verdict.textContent = "running the same query down both paths…";
    let report;
    try {
      report = await post("/api/v1/qdrant/bakeoff",
                          { query, k: 5, collection: "episodic", mode: "hybrid" }, 8000);
    } catch (err) {
      verdict.textContent = "the node did not answer — nothing to compare";
      btn.disabled = false;
      return;
    }
    const worst = Math.max(report.local_ms, report.native_ms) || 1;
    const lanes = [["qdLaneIdx", "qdMsIdx", report.local_ms],
                   ["qdLaneEng", "qdMsEng", report.native_ms]];
    lanes.forEach(([bar, num, ms]) => {
      el(bar).style.width = Math.max(2, Math.round((ms / worst) * 100)) + "%";
      el(num).textContent = ms.toFixed(2) + " ms";
    });
    if (report.plan && !report.plan.fell_back) {
      state.plan = report.plan;
      draw();
      caption();
    }
    const faster = report.native_ms < report.local_ms;
    verdict.className = "qd__verdict" + (faster ? " ok" : "");
    verdict.innerHTML =
      `top-k overlap <b>${report.overlap.toFixed(2)}</b> · rank agreement ` +
      `<b>${report.rank_agreement.toFixed(2)}</b> · ` +
      (faster ? "the engine won this query" : "the local index won this query") +
      `. ${report.plan && report.plan.fell_back
            ? "the engine declined: " + escapeHtml(report.plan.fell_back)
            : "both answered; whatever differs is the fusion constant, which "
              + "scripts/qdrant_bakeoff.py attributes over a whole corpus."}`;
    btn.disabled = false;
  }

  el("qdRaceBtn").addEventListener("click", () => {
    const typed = (document.getElementById("searchInput") || {}).value || "";
    race(typed.trim() || "coolant pressure alarm");
  });

  document.addEventListener("aegis:search", (e) => {
    // the plan for the query that just ran, read back from the node
    setTimeout(refresh, 120);
    const q = e.detail && e.detail.query;
    if (q) el("qdLastQuery").textContent = q;
  });

  refresh();
  setInterval(refresh, 6000);
})();
