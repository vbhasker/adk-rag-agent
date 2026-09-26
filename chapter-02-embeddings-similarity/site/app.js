"use strict";
// Chapter 2 playground. Edit this file, then run `npx tsc -p site` to regenerate app.js.
const $ = (id) => document.getElementById(id);
const escapeHtml = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c] ?? c);
async function api(path, body = {}) {
    const res = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await res.json();
    if (!res.ok)
        throw new Error(data.error ?? res.statusText);
    return data;
}
function setStatus(msg, isError = false) {
    const el = $("status");
    el.textContent = msg;
    el.classList.toggle("error", isError);
}
const CATEGORY_COLORS = {
    product: "#2563eb", support: "#16a34a", troubleshooting: "#dc2626", policy: "#d97706",
};
async function loadInfo() {
    try {
        const info = await api("/api/info");
        $("info").innerHTML = `Embedder: <code>${escapeHtml(info.embedder)}</code> · ${info.dim} dimensions.` +
            (info.isToy ? ` <strong style="color:var(--bad)">Toy hashing embedder active: results are NOT semantic.</strong>` : "");
    }
    catch {
        $("info").textContent = "Server not reachable. Start it to use the playground.";
    }
}
async function compare() {
    try {
        const r = await api("/api/compare", { a: $("a").value, b: $("b").value });
        const m = r.measures;
        const vec = (xs) => xs.map((x) => `<span class="token">${x.toFixed(3)}</span>`).join("") + ` <span class="muted">… (${r.dim} total)</span>`;
        $("cmp-out").innerHTML = `
      <div class="grid-2" style="margin-bottom:8px">
        <div class="mini"><h4>Cosine ${m.cosine.toFixed(4)}</h4><div class="bar"><span style="width:${Math.max(0, m.cosine) * 100}%"></span></div></div>
        <div class="mini"><h4>Dot ${m.dot.toFixed(4)}</h4><p class="muted">= cosine, because vectors have length 1</p></div>
        <div class="mini"><h4>Euclidean ${m.euclidean.toFixed(4)}</h4><p class="muted">√(2 − 2·cos) = ${Math.sqrt(Math.max(0, 2 - 2 * m.cosine)).toFixed(4)}</p></div>
        <div class="mini"><h4>Manhattan ${m.manhattan.toFixed(4)}</h4><p class="muted">no neat shortcut; can reorder results</p></div>
      </div>
      <div><strong>A:</strong> ${vec(r.previewA)}</div><div><strong>B:</strong> ${vec(r.previewB)}</div>`;
    }
    catch (e) {
        $("cmp-out").innerHTML = `<span class="status error">${escapeHtml(e.message)}</span>`;
    }
}
function column(title, hits, decimals) {
    const max = Math.max(...hits.map((h) => Math.abs(h.score)), 1e-9);
    const body = hits.length
        ? hits.map((h, i) => `
        <article class="result">
          <header><span class="title">#${i + 1} ${escapeHtml(h.title)}</span><span class="score">${h.score.toFixed(decimals)}</span></header>
          <div class="bar"><span style="width:${(100 * Math.abs(h.score)) / max}%"></span></div>
          <div class="small muted">${escapeHtml(h.category)}</div>
        </article>`).join("")
        : `<p class="muted">No results: no shared tokens.</p>`;
    return `<div><h4>${title}</h4><div class="results">${body}</div></div>`;
}
async function search() {
    setStatus("Embedding query and searching…");
    try {
        const query = $("q").value;
        const r = await api("/api/search", { query, metric: $("metric").value, topK: 5 });
        $("columns").innerHTML = column("Semantic (this chapter)", r.semantic, 4) + column("BM25 (Chapter 1)", r.bm25, 3);
        setStatus("");
        await drawMap(query);
    }
    catch (e) {
        setStatus(`Could not reach the API (${e.message}). Run: python server.py`, true);
    }
}
async function drawMap(query) {
    const r = await api("/api/map", { query });
    const all = [...r.docs, r.query];
    const xs = all.map((p) => p.x), ys = all.map((p) => p.y);
    const [minX, maxX, minY, maxY] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const W = 760, H = 380, pad = 40;
    const sx = (x) => pad + ((x - minX) / (maxX - minX || 1)) * (W - 2 * pad - 140);
    const sy = (y) => H - pad - ((y - minY) / (maxY - minY || 1)) * (H - 2 * pad);
    const best = [...r.docs].sort((a, b) => b.sim - a.sim).slice(0, 3).map((d) => d.id);
    const qx = sx(r.query.x), qy = sy(r.query.y);
    const lines = r.docs.filter((d) => best.includes(d.id))
        .map((d) => `<line x1="${qx}" y1="${qy}" x2="${sx(d.x)}" y2="${sy(d.y)}" stroke="var(--accent)" stroke-dasharray="4 4" opacity=".7" />`).join("");
    const dots = r.docs.map((d) => `
      <circle cx="${sx(d.x)}" cy="${sy(d.y)}" r="${best.includes(d.id) ? 8 : 6}" fill="${CATEGORY_COLORS[d.category] ?? "gray"}"><title>${escapeHtml(d.title)} · sim ${d.sim}</title></circle>
      <text x="${sx(d.x) + 10}" y="${sy(d.y) + 4}" font-size="11">${escapeHtml(d.id)}</text>`).join("");
    const legend = Object.entries(CATEGORY_COLORS)
        .map(([c, color], i) => `<circle cx="${W - 120}" cy="${30 + i * 20}" r="6" fill="${color}" /><text x="${W - 108}" y="${34 + i * 20}" font-size="12">${c}</text>`).join("");
    $("map").innerHTML = `
    <svg class="diagram" viewBox="0 0 ${W} ${H}" role="img" aria-label="2D PCA map of document embeddings and the query">
      <rect x="0" y="0" width="${W}" height="${H}" rx="10" fill="var(--surface-2)" />
      ${lines}${dots}
      <text x="${qx}" y="${qy + 7}" text-anchor="middle" font-size="22" fill="var(--accent)">★</text>
      <text x="${qx + 14}" y="${qy - 8}" font-weight="700" fill="var(--accent)">${escapeHtml(r.query.text.slice(0, 40))}</text>
      ${legend}
    </svg>`;
}
function chips(containerId, items, onPick) {
    const el = $(containerId);
    el.innerHTML = items.map((q) => `<button class="chip" type="button">${escapeHtml(q)}</button>`).join("");
    el.addEventListener("click", (ev) => {
        const t = ev.target;
        if (t.classList.contains("chip"))
            onPick(t.textContent ?? "");
    });
}
function init() {
    chips("pairs", ["bike | bicycle", "bike | banana", "I love my bike | I do not love my bike", "battery | Batterie", "E-07 | motor overheating", "can I get my money back | refund policy"], (pair) => {
        const [a, b] = pair.split(" | ");
        $("a").value = a;
        $("b").value = b;
        void compare();
    });
    chips("examples", ["how long does the battery last", "my bike won't turn on", "can I send it back if I don't like it", "E07", "UN3480", "is a speed chip covered?", "kids on the back"], (q) => {
        $("q").value = q;
        void search();
    });
    $("cmp").addEventListener("click", () => void compare());
    $("go").addEventListener("click", () => void search());
    $("metric").addEventListener("change", () => void search());
    $("q").addEventListener("keydown", (ev) => { if (ev.key === "Enter")
        void search(); });
    void loadInfo();
    void compare();
    void search();
}
init();
