"use strict";
// Chapter 3 playground. Edit this file, then run `npx -p typescript tsc -p site` to regenerate app.js.
const $ = (id) => document.getElementById(id);
const value = (id) => $(id).value;
const escapeHtml = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c] ?? c);
async function api(path, body = {}) {
    const res = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await res.json();
    if (!res.ok)
        throw new Error(data.error ?? res.statusText);
    return data;
}
function setStatus(id, msg, isError = false) {
    $(id).textContent = msg;
    $(id).classList.toggle("error", isError);
}
const PALETTE = ["#0e7490", "#7c3aed", "#c2410c", "#15803d", "#be185d", "#a16207"];
const chunkSettings = () => ({ strategy: value("strategy"), size: Number(value("size")), overlap: Number(value("overlap")) });
async function loadInfo() {
    try {
        const info = await api("/api/info");
        $("info").innerHTML = `Embedder <code>${escapeHtml(info.embedder)}</code> · ${info.chunks} chunks indexed · LLM: ${info.llm ? "✅ Gemini configured" : "❌ no key (search works, agent won't)"}`;
        $("doc").innerHTML = info.docs.map((d) => `<option value="${d.id}" ${d.id === "warranty" ? "selected" : ""}>${escapeHtml(d.title)}</option>`).join("");
        await visualize();
    }
    catch {
        $("info").textContent = "Server not reachable. Start it to use the playground.";
    }
}
async function visualize() {
    const r = await api("/api/chunk", { docId: value("doc"), ...chunkSettings() });
    const avg = r.chunks.reduce((n, c) => n + c.words, 0) / Math.max(1, r.chunks.length);
    setStatus("chunk-summary", `${r.chunks.length} chunks · avg ${avg.toFixed(0)} words`);
    $("chunks").innerHTML = r.chunks.map((c, i) => `
    <div class="chunk" style="--c:${PALETTE[i % PALETTE.length]}">
      <div class="head">[${escapeHtml(c.id)}] ${c.section ? `§ ${escapeHtml(c.section)} · ` : ""}${c.words} words</div>${escapeHtml(c.text)}</div>`).join("");
}
function renderHits(hits) {
    const max = Math.max(...hits.map((h) => h.score), 1e-9);
    return hits.map((h, i) => `
    <article class="result">
      <header><span class="title">#${i + 1} [${escapeHtml(h.source_id)}] ${escapeHtml(h.title)}${h.section ? ` › ${escapeHtml(h.section)}` : ""}</span><span class="score">${h.score.toFixed(4)}</span></header>
      <div class="bar"><span style="width:${(100 * Math.max(0, h.score)) / max}%"></span></div>
      <div class="snippet">${escapeHtml(h.text)}</div>
    </article>`).join("");
}
async function search() {
    setStatus("status", "Searching…");
    try {
        const r = await api("/api/search", { query: value("q"), category: value("category"), topK: 4, ...chunkSettings() });
        setStatus("status", `Top ${r.results.length} of ${r.chunkCount} chunks (${chunkSettings().strategy}, ${chunkSettings().size} words)`);
        $("results").innerHTML = renderHits(r.results);
    }
    catch (e) {
        setStatus("status", `Could not reach the API (${e.message}). Run: python server.py`, true);
    }
}
function renderTrace(trace) {
    return trace.filter((s) => s.type !== "message").map((s) => s.type === "tool_call"
        ? `<div>🔎 <strong>${escapeHtml(s.author)}</strong> → ${escapeHtml(s.name ?? "")}(${escapeHtml(JSON.stringify(s.args))})</div>`
        : `<div>&nbsp;&nbsp;↳ ${escapeHtml((s.response?.results ?? []).map((h) => h.source_id).join(", ") || JSON.stringify(s.response).slice(0, 160))}</div>`).join("");
}
async function askAgent() {
    setStatus("ask-status", "Thinking… (the agent may search more than once)");
    $("ask").setAttribute("disabled", "");
    try {
        const r = await api("/api/ask", { question: value("question") });
        if (r.error) {
            setStatus("ask-status", r.error, true);
            return;
        }
        setStatus("ask-status", "");
        $("answer").innerHTML = `<div class="trace">${renderTrace(r.trace ?? [])}</div><div class="answer">${escapeHtml(r.answer ?? "")}</div>`;
    }
    catch (e) {
        setStatus("ask-status", e.message, true);
    }
    finally {
        $("ask").removeAttribute("disabled");
    }
}
function init() {
    const examples = ["is a speed chip covered by the warranty?", "what does E-12 mean", "how do I store the battery in winter", "UN3480", "refund", "kids on the back of the bike"];
    $("examples").innerHTML = examples.map((q) => `<button class="chip" type="button">${escapeHtml(q)}</button>`).join("");
    $("examples").addEventListener("click", (ev) => {
        const t = ev.target;
        if (t.classList.contains("chip")) {
            $("q").value = t.textContent ?? "";
            void search();
        }
    });
    for (const id of ["size", "overlap"]) {
        $(id).addEventListener("input", () => { $(`${id}v`).textContent = value(id); void visualize(); });
    }
    for (const id of ["doc", "strategy"])
        $(id).addEventListener("change", () => void visualize());
    $("category").addEventListener("change", () => void search());
    $("go").addEventListener("click", () => void search());
    $("q").addEventListener("keydown", (ev) => { if (ev.key === "Enter")
        void search(); });
    $("ask").addEventListener("click", () => void askAgent());
    void loadInfo();
    void search();
}
init();
