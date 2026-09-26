"use strict";
// Chapter 7 playground. Edit this file, then run `npx tsc -p site` to regenerate app.js.
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
// ---------- Metric calculator (pure TypeScript, mirrors metrics.py) ----------
const hits = new Set([1, 3]);
function renderCalculator() {
    const nRel = Number(value("calc-nrel"));
    $("calc-ranks").innerHTML = [1, 2, 3, 4, 5].map((r) => `<button type="button" data-rank="${r}" class="${hits.has(r) ? "on" : ""}">#${r} ${hits.has(r) ? "✓ relevant" : "✗"}</button>`).join("");
    const found = [...hits].filter((r) => r <= 5).sort((a, b) => a - b).slice(0, nRel);
    const k = 5;
    const dcg = found.reduce((s, r) => s + 1 / Math.log2(r + 1), 0);
    const ideal = Array.from({ length: Math.min(nRel, k) }, (_, i) => 1 / Math.log2(i + 2)).reduce((a, b) => a + b, 0);
    const metrics = [
        ["Hit@5", found.length ? 1 : 0],
        ["Recall@5", found.length / nRel],
        ["Precision@5", found.length / k],
        ["MRR", found.length ? 1 / found[0] : 0],
        ["nDCG@5", ideal ? dcg / ideal : 0],
    ];
    $("calc-out").innerHTML = metrics.map(([m, v]) => `<span class="token">${m} = ${v.toFixed(3)}</span>`).join(" ") +
        (hits.size > nRel ? ` <span class="muted">(only the first ${nRel} ticks count: there are only ${nRel} relevant docs)</span>` : "");
}
// ---------- Leaderboard ----------
let report = null;
function barChart(metric) {
    if (!report)
        return "";
    const entries = Object.entries(report.configs).map(([name, r]) => [name, r.summary[metric]]).sort((a, b) => b[1] - a[1]);
    return entries.map(([name, v]) => `
    <div class="bar-row" title="${escapeHtml(name)}: ${metric} = ${v.toFixed(3)}">
      <span class="label">${escapeHtml(name)}</span>
      <span class="track"><span class="fill" style="display:block;width:${Math.max(1, v * 100)}%"></span></span>
      <span class="val">${v.toFixed(3)}</span>
    </div>`).join("");
}
function renderLeaderboard() {
    if (!report)
        return;
    const k = report.k;
    const cols = [`hit@${k}`, `recall@${k}`, "mrr", `ndcg@${k}`, `precision@${k}`];
    $("mrr-title").textContent = "MRR (higher is better)";
    $("recall-title").textContent = `Recall@${k} (higher is better)`;
    $("chart-mrr").innerHTML = barChart("mrr");
    $("chart-recall").innerHTML = barChart(`recall@${k}`);
    const best = Object.fromEntries(cols.map((c) => [c, Math.max(...Object.values(report.configs).map((r) => r.summary[c]))]));
    const rows = Object.entries(report.configs).map(([name, r]) => `<tr><td>${escapeHtml(name)}</td>${cols.map((c) => `<td${r.summary[c] === best[c] ? ' class="ok"' : ""}>${r.summary[c].toFixed(3)}</td>`).join("")}<td>${r.msPerQuery}</td></tr>`).join("");
    $("leaderboard").innerHTML = `<div class="table-wrap"><table><tr><th>Configuration</th>${cols.map((c) => `<th>${c}</th>`).join("")}<th>ms / query</th></tr>${rows}</table></div>`;
    $("drill").innerHTML = Object.keys(report.configs).map((n) => `<option>${escapeHtml(n)}</option>`).join("");
    renderFailures();
}
function renderFailures() {
    if (!report)
        return;
    const cfg = report.configs[value("drill")];
    if (!cfg)
        return;
    const k = report.k;
    const weak = cfg.rows.filter((r) => r[`recall@${k}`] < 1).sort((a, b) => a.mrr - b.mrr);
    $("failures").innerHTML = weak.length
        ? `<div class="table-wrap"><table><tr><th>Question</th><th>Needed</th><th>Got (top ${k} docs)</th><th>MRR</th></tr>${weak.map((r) => `
        <tr><td>${escapeHtml(r.question)}<br /><span class="small muted">${escapeHtml(r.type)}</span></td>
        <td>${r.relevant.map(escapeHtml).join(", ")}</td>
        <td>${r.ranked.map((d) => `<span class="${r.relevant.includes(d) ? "ok" : ""}">${escapeHtml(d)}</span>`).join(", ")}</td>
        <td class="${r.mrr ? "" : "miss"}">${r.mrr.toFixed(2)}</td></tr>`).join("")}</table></div>`
        : `<p class="ok">Perfect recall@${k} on every question. Time to write harder questions!</p>`;
}
async function runRetrievalEval() {
    setStatus("status", "Evaluating all configurations over the golden set…");
    $("run").setAttribute("disabled", "");
    try {
        report = await api("/api/eval-retrieval", { k: Number(value("k")), llm: $("useLlm").checked });
        setStatus("status", `${report.questions} answerable questions · k=${report.k} · embedder ${report.embedder} · reranker ${report.reranker}`);
        renderLeaderboard();
    }
    catch (e) {
        setStatus("status", `Could not reach the API (${e.message}). Run: python server.py`, true);
    }
    finally {
        $("run").removeAttribute("disabled");
    }
}
// ---------- Answer eval ----------
async function loadAnswers(run) {
    setStatus("answers-status", run ? "Running the agent + judge on every question. This takes a few minutes…" : "Loading…");
    try {
        const r = await api("/api/answers", { run });
        if (r.error) {
            setStatus("answers-status", r.error, true);
            return;
        }
        setStatus("answers-status", `Model: ${r.model}`);
        const summary = Object.entries(r.summary ?? {}).map(([k, v]) => `<span class="token">${escapeHtml(k)}: ${escapeHtml(String(v))}</span>`).join(" ");
        const rows = (r.rows ?? []).map((x) => `<tr><td>${escapeHtml(x.question)}<br /><span class="small muted">${escapeHtml(x.type)}</span></td>
      <td>${x.faithfulness.toFixed(2)}</td><td>${x.correctness.toFixed(2)}</td><td>${x.refused ? "yes" : "no"}</td>
      <td class="${x.citations_ok ? "ok" : "miss"}">${x.citations_ok ? "✓" : "✗"}</td><td class="small">${escapeHtml(x.explanation)}</td></tr>`).join("");
        $("answers").innerHTML = `<div style="margin-bottom:10px">${summary}</div><div class="table-wrap"><table><tr><th>Question</th><th>Faithful</th><th>Correct</th><th>Refused</th><th>Cites</th><th>Judge says</th></tr>${rows}</table></div>`;
    }
    catch (e) {
        setStatus("answers-status", e.message, true);
    }
}
async function loadGolden() {
    try {
        const golden = await api("/api/golden");
        $("golden").innerHTML = `<details><summary>Show all ${golden.length} golden questions</summary><div class="table-wrap"><table><tr><th>id</th><th>type</th><th>question</th><th>relevant docs</th></tr>${golden.map((g) => `<tr><td>${g.id}</td><td>${escapeHtml(g.type)}</td><td>${escapeHtml(g.question)}</td><td>${g.relevant_docs.map(escapeHtml).join(", ") || "<em>none</em>"}</td></tr>`).join("")}</table></div></details>`;
    }
    catch {
        /* static view: nothing to show */
    }
}
async function loadInfo() {
    try {
        const info = await api("/api/info");
        $("info").innerHTML = `Embedder <code>${escapeHtml(info.embedder)}</code> · reranker <code>${escapeHtml(info.reranker)}</code> · Gemini ${info.llm ? "✅" : "❌"}`;
    }
    catch {
        $("info").textContent = "Server not reachable.";
    }
}
function init() {
    $("calc-ranks").addEventListener("click", (ev) => {
        const btn = ev.target.closest("button");
        if (!btn)
            return;
        const r = Number(btn.dataset.rank);
        if (hits.has(r))
            hits.delete(r);
        else
            hits.add(r);
        renderCalculator();
    });
    $("calc-nrel").addEventListener("change", renderCalculator);
    renderCalculator();
    $("run").addEventListener("click", () => void runRetrievalEval());
    $("k").addEventListener("change", () => void runRetrievalEval());
    $("drill").addEventListener("change", renderFailures);
    $("load-answers").addEventListener("click", () => void loadAnswers(false));
    $("run-answers").addEventListener("click", () => void loadAnswers(true));
    void loadInfo();
    void loadGolden();
    void runRetrievalEval();
}
init();
