"use strict";
// Chapter 6 playground. Edit this file, then run `npx -p typescript tsc -p site` to regenerate app.js.
const $ = (id) => document.getElementById(id);
const value = (id) => $(id).value;
const checked = (id) => $(id).checked;
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
function card(h, i) {
    return `
    <article class="result">
      <header><span class="title">#${i + 1} ${escapeHtml(h.source_id)}</span><span class="score">${(h.rerank_score ?? h.score).toFixed(4)}</span></header>
      <div class="small">${escapeHtml(h.title)} › ${escapeHtml(h.section)}</div>
      ${h.matched_child ? `<div class="child">matched child: “${escapeHtml(h.matched_child)}”</div>` : ""}
      <div class="snippet">${escapeHtml(h.text.slice(0, 180))}…</div>
    </article>`;
}
const variantLines = () => value("queries").split("\n").map((l) => l.trim()).filter(Boolean);
async function generate() {
    setStatus("status", "Generating query variants…");
    try {
        const r = await api("/api/transform", { query: value("q"), hyde: checked("useHyde") });
        $("queries").value = r.multi.queries.join("\n");
        if (r.hyde)
            $("hydePassage").value = r.hyde.passage ?? "";
        setStatus("status", `Variants: ${r.multi.source}${r.hyde ? ` · HyDE: ${r.hyde.source}` : ""}`);
    }
    catch (e) {
        setStatus("status", e.message, true);
    }
}
async function search() {
    setStatus("status", "Searching…");
    try {
        const queries = variantLines();
        const r = await api("/api/search", {
            query: value("q"),
            queries: queries.length ? queries : [value("q")],
            hydePassage: value("hydePassage").trim(),
            contextMode: value("contextMode"),
            parentDocs: checked("parentDocs"),
            rerank: checked("rerank"),
            topK: 5,
        });
        setStatus("status", `${r.queries.length} quer${r.queries.length === 1 ? "y" : "ies"}${value("hydePassage").trim() ? " + HyDE" : ""} · context ${r.contextMode} · reranker ${r.reranker} · ${Object.entries(r.timings).map(([k, v]) => `${k.replace("_ms", "")} ${v} ms`).join(" · ")}`);
        const cols = [["Searched with", `<div class="results">${r.queries.map((q) => `<div class="token">${escapeHtml(q)}</div>`).join("")}</div>`],
            ["Top 5 after fusion + rerank", `<div class="results">${r.final.map(card).join("")}</div>`]];
        $("columns").innerHTML = cols.map(([t, body]) => `<div><h4>${t}</h4>${body}</div>`).join("");
    }
    catch (e) {
        setStatus("status", `Could not reach the API (${e.message}). Run: python server.py`, true);
    }
}
async function compareContext() {
    try {
        const r = await api("/api/compare-context", { query: value("cq") });
        $("ccolumns").innerHTML = Object.entries(r).map(([mode, hits]) => `<div><h4>context: ${escapeHtml(mode)}</h4><div class="results">${hits.map(card).join("")}</div></div>`).join("");
    }
    catch (e) {
        $("ccolumns").innerHTML = `<p class="status error">${escapeHtml(e.message)}</p>`;
    }
}
async function askAgent() {
    setStatus("ask-status", "Thinking…");
    $("ask").setAttribute("disabled", "");
    try {
        const r = await api("/api/ask", { question: value("question") });
        if (r.error) {
            setStatus("ask-status", r.error, true);
            return;
        }
        setStatus("ask-status", "");
        const trace = (r.trace ?? []).filter((s) => s.type !== "message").map((s) => s.type === "tool_call"
            ? `<div>🔎 ${escapeHtml(s.name ?? "")}(${escapeHtml(JSON.stringify(s.args))})</div>`
            : `<div>&nbsp;&nbsp;↳ ${escapeHtml((s.response?.results ?? []).map((h) => h.source_id).join(", "))}</div>`).join("");
        $("answer").innerHTML = `<div class="trace">${trace}</div><div class="answer">${escapeHtml(r.answer ?? "")}</div>`;
    }
    catch (e) {
        setStatus("ask-status", e.message, true);
    }
    finally {
        $("ask").removeAttribute("disabled");
    }
}
async function loadInfo() {
    try {
        const info = await api("/api/info");
        $("info").innerHTML = `Embedder <code>${escapeHtml(info.embedder)}</code> · reranker <code>${escapeHtml(info.reranker)}</code> · LLM ${info.llm ? "✅" : "❌ (heuristic fallbacks; LLM context unavailable)"}`;
        $("contextMode").innerHTML = info.contextModes.map((m) => `<option ${m === info.defaultContextMode ? "selected" : ""}>${m}</option>`).join("");
    }
    catch {
        $("info").textContent = "Server not reachable.";
    }
}
function init() {
    const examples = ["can my kid ride on the back and is it legal in the EU?", "how do I clear stored codes", "it's dead, nothing on the screen", "cheapest way to carry groceries and how far can it go"];
    $("examples").innerHTML = examples.map((q) => `<button class="chip" type="button">${escapeHtml(q)}</button>`).join("");
    $("examples").addEventListener("click", (ev) => {
        const t = ev.target;
        if (!t.classList.contains("chip"))
            return;
        $("q").value = t.textContent ?? "";
        $("queries").value = "";
        $("hydePassage").value = "";
        void search();
    });
    $("gen").addEventListener("click", () => void generate());
    $("go").addEventListener("click", () => void search());
    $("q").addEventListener("keydown", (ev) => { if (ev.key === "Enter")
        void search(); });
    for (const id of ["contextMode", "parentDocs", "rerank"])
        $(id).addEventListener("change", () => void search());
    $("cgo").addEventListener("click", () => void compareContext());
    $("ask").addEventListener("click", () => void askAgent());
    void loadInfo().then(() => { void search(); void compareContext(); });
}
init();
