"use strict";
// Chapter 1 playground. Edit this file, then run `npx tsc -p site` to regenerate app.js.
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
const tokens = (list, hits = new Set()) => list.length ? list.map((t) => `<span class="token${hits.has(t) ? " hit" : ""}">${escapeHtml(t)}</span>`).join("") : "<em>(none)</em>";
async function runTokenizer() {
    try {
        const r = await api("/api/tokenize", { text: $("tok-input").value });
        $("tok-out").innerHTML = `
      <div><strong>1 · split + lowercase:</strong> ${tokens(r.raw)}</div>
      <div><strong>2 · minus stop words:</strong> ${tokens(r.noStopWords)}</div>
      <div><strong>3 · stemmed:</strong> ${tokens(r.stemmed)}</div>`;
    }
    catch {
        $("tok-out").innerHTML = `<span class="muted">Start <code>python server.py</code> to try the tokenizer live.</span>`;
    }
}
async function runSearch() {
    setStatus("Searching…");
    try {
        const r = await api("/api/search", {
            query: $("q").value,
            algo: $("algo").value,
            k1: Number($("k1").value),
            b: Number($("b").value),
            stem: $("stem").checked,
            stopWords: $("stop").checked,
            topK: 6,
        });
        const idfRows = Object.entries(r.idf)
            .map(([t, s]) => `<span class="token${s.df ? " hit" : ""}" title="in ${s.df} docs">${escapeHtml(t)} · idf ${s.idf} · df ${s.df}</span>`)
            .join("");
        $("query-info").innerHTML = `Query tokens → ${idfRows || "<em>(none)</em>"} <span class="muted">· avg doc length ${r.avgdl} tokens</span>`;
        setStatus(r.results.length ? `${r.results.length} matching documents` : "No document shares a single token with your query. Keyword search is blind here.");
        const max = Math.max(...r.results.map((x) => x.score), 1e-9);
        $("results").innerHTML = r.results
            .map((x, i) => `
        <article class="result">
          <header><span class="title">#${i + 1} ${escapeHtml(x.title)}</span><span class="score">${x.score.toFixed(3)}</span></header>
          <div class="bar"><span style="width:${(100 * x.score) / max}%"></span></div>
          <div class="breakdown">${Object.entries(x.breakdown)
            .sort((a, b) => b[1] - a[1])
            .map(([t, v]) => `<span>${escapeHtml(t)} +${v.toFixed(2)}</span>`)
            .join("")}<span class="muted">${x.length} tokens · ${escapeHtml(x.category)}</span></div>
          <div class="snippet">${escapeHtml(x.snippet)}…</div>
        </article>`)
            .join("");
    }
    catch (e) {
        setStatus(`Could not reach the API (${e.message}). Run: python server.py`, true);
    }
}
async function loadStats() {
    try {
        const s = await api("/api/stats");
        const fmt = (xs) => xs.map((t) => `<span class="token">${escapeHtml(t.term)} · df ${t.df} · idf ${t.idf}</span>`).join("");
        $("stats").innerHTML = `${s.documents} documents · ${s.vocabulary} unique tokens · average ${s.avgdl} tokens per doc
      <div style="margin-top:8px"><strong>Most common (lowest IDF):</strong> ${fmt(s.commonTerms)}</div>
      <div><strong>Rarest (highest IDF):</strong> ${fmt(s.rareTerms)}</div>`;
    }
    catch {
        $("stats").textContent = "Start the server to see live corpus statistics.";
    }
}
function init() {
    const examples = ["error E-07", "E07", "how long does the battery last", "refund", "battery", "charging", "my bike won't turn on", "UN3480"];
    $("examples").innerHTML = examples.map((q) => `<button class="chip" type="button">${escapeHtml(q)}</button>`).join("");
    $("examples").addEventListener("click", (ev) => {
        const target = ev.target;
        if (target.classList.contains("chip")) {
            $("q").value = target.textContent ?? "";
            void runSearch();
        }
    });
    for (const id of ["k1", "b"]) {
        $(id).addEventListener("input", () => {
            $(`${id}v`).textContent = $(id).value;
            void runSearch();
        });
    }
    for (const id of ["algo", "stem", "stop"])
        $(id).addEventListener("change", () => void runSearch());
    $("go").addEventListener("click", () => void runSearch());
    $("q").addEventListener("keydown", (ev) => { if (ev.key === "Enter")
        void runSearch(); });
    $("tok-btn").addEventListener("click", () => void runTokenizer());
    void runTokenizer();
    void runSearch();
    void loadStats();
}
init();
