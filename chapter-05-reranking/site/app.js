"use strict";
// Chapter 5 playground. Edit this file, then run `npx -p typescript tsc -p site` to regenerate app.js.
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
function movement(newRank, oldRank) {
    if (!oldRank)
        return "";
    const delta = oldRank - newRank;
    if (delta > 0)
        return `<span class="move up">↑${delta} (was #${oldRank})</span>`;
    if (delta < 0)
        return `<span class="move down">↓${-delta} (was #${oldRank})</span>`;
    return `<span class="move same">= #${oldRank}</span>`;
}
function card(h, i, showMove) {
    const score = h.rerank_score ?? h.score;
    return `
    <article class="result">
      <header><span class="title">#${i + 1} ${escapeHtml(h.source_id)}</span><span class="score">${score.toFixed(4)}</span></header>
      <div class="small">${escapeHtml(h.section || h.title)} ${showMove ? movement(i + 1, h.fused_rank) : ""}</div>
      <div class="snippet">${escapeHtml(h.text.slice(0, 160))}…</div>
    </article>`;
}
async function search() {
    setStatus("status", "Retrieving and reranking…");
    const minScore = Number(value("minScore"));
    try {
        const r = await api("/api/search", {
            query: value("q"),
            reranker: value("reranker"),
            candidates: Number(value("candidates")),
            category: value("category"),
            mmr: checked("mmr"),
            mmrLambda: Number(value("mmrLambda")),
            minScore: minScore > 0 ? minScore : null,
            topK: 5,
        });
        const t = r.timings;
        setStatus("status", `Reranker: ${r.reranker} · hybrid ${t.hybrid_ms ?? 0} ms${t.rerank_ms !== undefined ? ` · rerank ${t.rerank_ms} ms` : ""}`);
        const after = r.final.length ? r.final.map((h, i) => card(h, i, true)).join("") : `<p class="muted">Nothing passed the threshold. The agent would now say "I don't know".</p>`;
        $("columns").innerHTML = `
      <div><h4>Before · hybrid order</h4><div class="results">${r.fused.map((h, i) => card(h, i, false)).join("")}</div></div>
      <div><h4>After · ${escapeHtml(r.reranker)}${checked("mmr") ? " + MMR" : ""}</h4><div class="results">${after}</div></div>`;
    }
    catch (e) {
        setStatus("status", `Could not reach the API (${e.message}). Run: python server.py`, true);
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
            : `<div>&nbsp;&nbsp;↳ ${escapeHtml((s.response?.results ?? []).map((h) => `${h.source_id} (${(h.rerank_score ?? h.score).toFixed(2)})`).join(", "))}</div>`).join("");
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
        $("info").innerHTML = `Embedder <code>${escapeHtml(info.embedder)}</code> · default reranker <code>${escapeHtml(info.defaultReranker)}</code> · LLM ${info.llm ? "✅" : "❌"}`;
        const best = info.rerankers.includes("cross-encoder") ? "cross-encoder" : info.rerankers.includes("gemini") ? "gemini" : "none";
        $("reranker").innerHTML = ["off", ...info.rerankers].map((k) => `<option ${k === best ? "selected" : ""}>${k}</option>`).join("");
        if (!info.rerankers.includes("cross-encoder")) {
            $("info").innerHTML += ` <strong style="color:var(--bad)">Cross-encoder couldn't load (see server log). Reranking falls back to ${best}.</strong>`;
        }
    }
    catch {
        $("info").textContent = "Server not reachable.";
    }
}
function init() {
    const examples = ["what happens if the motor gets too hot", "is a speed chip covered by the warranty?", "warranty", "do you sell electric scooters?", "can I ride in the rain and wash the bike with a hose", "how long does the battery last"];
    $("examples").innerHTML = examples.map((q) => `<button class="chip" type="button">${escapeHtml(q)}</button>`).join("");
    $("examples").addEventListener("click", (ev) => {
        const t = ev.target;
        if (t.classList.contains("chip")) {
            $("q").value = t.textContent ?? "";
            void search();
        }
    });
    $("mmrLambda").addEventListener("input", () => { $("mmrLambdav").textContent = value("mmrLambda"); });
    $("minScore").addEventListener("input", () => { $("minScorev").textContent = Number(value("minScore")) > 0 ? value("minScore") : "off"; });
    for (const id of ["reranker", "candidates", "category", "mmr", "mmrLambda", "minScore"])
        $(id).addEventListener("change", () => void search());
    $("go").addEventListener("click", () => void search());
    $("q").addEventListener("keydown", (ev) => { if (ev.key === "Enter")
        void search(); });
    $("ask").addEventListener("click", () => void askAgent());
    void loadInfo().then(search);
}
init();
