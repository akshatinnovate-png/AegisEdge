/* ============================================================
   AegisEdge — what the link would carry, and why

   The queue is durable, so nothing is lost either way. What this panel
   shows is the part that would arrive *first* if the link dropped now,
   and the reason each operation earned its place. Every row is priced
   by the same planner a real cycle uses; asking for this plan does not
   send anything.

   The control is drawn beside it on purpose. "Value-first is better"
   is only a claim worth making next to what write-order would have
   carried in the same bytes.
   ============================================================ */

(() => {
  const el = (id) => document.getElementById(id);
  const panel = el("egPanel");
  if (!panel) return;

  const KINDS = { delete: "OBLIGATION", upsert: "UPDATE" };

  function esc(text) {
    const div = document.createElement("div");
    div.textContent = String(text == null ? "" : text);
    return div.innerHTML;
  }

  function render(data) {
    const rows = el("egRows");
    rows.textContent = "";
    if (!data) {
      el("egLink").textContent = "—";
      el("egCaption").textContent = "node unreachable — no plan to show";
      el("egBar").style.width = "0%";
      return;
    }
    const plan = data.plan || {};
    const top = plan.top || [];

    el("egLink").textContent =
      `${data.link} · ${data.queued} queued · ${plan.send} would go now`;
    const budget = plan.budget_bytes || 0;
    el("egBar").style.width = budget
      ? Math.min(100, Math.round((plan.planned_bytes / budget) * 100)) + "%"
      : "100%";
    el("egBudget").textContent = budget
      ? `${(plan.planned_bytes / 1024).toFixed(1)} of ${(budget / 1024).toFixed(0)} KB this cycle`
      : `${(plan.planned_bytes / 1024).toFixed(1)} KB · no cap on a healthy link`;

    const peak = Math.max(...top.map((r) => r.value_per_kb), 1);
    top.forEach((row, i) => {
      const line = document.createElement("div");
      line.className = "eg-row";
      line.style.animationDelay = i * 45 + "ms";
      line.innerHTML =
        `<span class="eg-kind eg-kind--${esc(row.kind)}">${KINDS[row.kind] || esc(row.kind)}</span>` +
        `<span class="eg-why"></span>` +
        `<i class="eg-meter"><b style="width:${Math.round((row.value_per_kb / peak) * 100)}%"></b></i>` +
        `<span class="eg-num">${row.value_per_kb.toFixed(0)}/KB</span>`;
      line.querySelector(".eg-why").textContent =
        row.reasons.length ? row.reasons[0] : "no signal beyond being queued";
      rows.appendChild(line);
    });
    if (!top.length) {
      const empty = document.createElement("div");
      empty.className = "eg-empty";
      empty.textContent = "the queue is empty — everything the fleet is owed has landed";
      rows.appendChild(empty);
    }

    const counters = data.counters || {};
    const gain = data.fifo_value > 0 ? data.value / data.fifo_value : 1;
    const tie = Math.abs(data.value - data.fifo_value) < 0.01;
    el("egCaption").innerHTML =
      (tie
        ? `the whole queue fits this link, so write order and value order carry the ` +
          `same <b>${data.value.toFixed(0)}</b> — ordering only decides anything once ` +
          `the link is too small for the queue`
        : `these bytes carry <b>${data.value.toFixed(0)}</b> of value; in write order ` +
          `the same bytes would carry <b>${data.fifo_value.toFixed(0)}</b> ` +
          `(<b>${gain.toFixed(1)}×</b>)`) +
      ` · <b>${counters.suppressed_as_redundant || 0}</b> superseded before they left, ` +
      `saving <b>${((counters.bytes_not_sent || 0) / 1024).toFixed(1)} KB</b> ` +
      `· nothing waits more than <b>${counters.starvation_ceiling_s || 0}s</b>`;
  }

  async function refresh() {
    try {
      render(await get("/api/v1/sync/egress", 4000));
    } catch {
      render(null);
    }
  }

  document.addEventListener("aegis:search", () => setTimeout(refresh, 200));
  el("egRefreshBtn").addEventListener("click", refresh);
  refresh();
  setInterval(refresh, 7000);
})();
