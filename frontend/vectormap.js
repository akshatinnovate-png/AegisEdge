/* ============================================================
   AegisEdge — the corpus, projected

   Drawn from vectors scrolled back out of Qdrant, not from a copy this
   page kept: a picture of the corpus sourced from the process drawing
   it would agree with itself whatever the engine actually stored.

   Two axes cannot hold a 256-dimensional space, so the share of
   variance they do hold is printed under the plot. Without it, someone
   reads adjacency on a screen as similarity in the space, which is the
   single most common way a vector visualisation misleads.
   ============================================================ */

(() => {
  const el = (id) => document.getElementById(id);
  const panel = el("vmPanel");
  if (!panel) return;

  const NS = "http://www.w3.org/2000/svg";
  const W = 460, H = 300, PAD = 16;
  let lastQuery = "";

  const px = (u) => PAD + u * (W - PAD * 2);
  const py = (u) => H - PAD - u * (H - PAD * 2);

  function node(name, attrs, cls) {
    const n = document.createElementNS(NS, name);
    Object.entries(attrs).forEach(([k, v]) => n.setAttribute(k, v));
    if (cls) n.setAttribute("class", cls);
    return n;
  }

  function render(data) {
    const svg = el("vmPlot");
    svg.textContent = "";
    if (!data || !data.points || !data.points.length) {
      el("vmCaption").textContent = data && data.detail
        ? data.detail
        : "node unreachable — nothing to plot";
      el("vmMeta").textContent = "—";
      return;
    }

    const hits = data.points.filter((p) => p.rank);
    const q = data.query;

    // lines from the query to what it retrieved, drawn first so the dots sit
    // on top of them rather than under
    if (q) {
      hits.forEach((p) => {
        svg.appendChild(node("line", {
          x1: px(q[0]), y1: py(q[1]), x2: px(p.xy[0]), y2: py(p.xy[1]),
        }, "vm-link"));
      });
    }

    data.points.forEach((p, i) => {
      const hit = !!p.rank;
      const dot = node("circle", {
        cx: px(p.xy[0]), cy: py(p.xy[1]), r: hit ? 4.6 : 2.1,
      }, "vm-dot" + (hit ? " vm-dot--hit" : "") + (p.stale ? " vm-dot--stale" : ""));
      dot.style.animationDelay = (i % 60) * 8 + "ms";
      const title = document.createElementNS(NS, "title");
      title.textContent = (hit ? `#${p.rank} · ` : "") + (p.text || p.id);
      dot.appendChild(title);
      svg.appendChild(dot);
      if (hit) {
        const rank = node("text", {
          x: px(p.xy[0]) + 7, y: py(p.xy[1]) + 3,
        }, "vm-rank");
        rank.textContent = p.rank;
        svg.appendChild(rank);
      }
    });

    if (q) {
      svg.appendChild(node("circle", { cx: px(q[0]), cy: py(q[1]), r: 9 }, "vm-qring"));
      svg.appendChild(node("circle", { cx: px(q[0]), cy: py(q[1]), r: 3 }, "vm-q"));
    }

    const share = (data.explained_variance * 100).toFixed(1);
    el("vmCaption").innerHTML =
      `these two axes carry <b>${share}%</b> of the variance in a ` +
      `${data.sampled}-point sample. Adjacency here is a hint about the space, ` +
      `not a measurement of it — the numbered hits are the nearest in 256 ` +
      `dimensions, which is why they are not the nearest on this page.`;
    el("vmMeta").textContent =
      `${data.sampled} points from ${data.backend}` +
      (data.query_text ? ` · ${data.matched} retrieved for "${data.query_text}"` : "");
  }

  async function refresh(query) {
    lastQuery = query != null ? query : lastQuery;
    const q = lastQuery ? "&query=" + encodeURIComponent(lastQuery) + "&k=5" : "";
    try {
      render(await get(`/api/v1/qdrant/map?collection=episodic&limit=400${q}`, 6000));
    } catch {
      render(null);
    }
  }

  document.addEventListener("aegis:search", (e) => {
    const typed = e.detail && e.detail.query;
    if (typed) refresh(typed);
  });
  el("vmReloadBtn").addEventListener("click", () => refresh());

  refresh("");
  setInterval(() => refresh(), 20000);
})();
