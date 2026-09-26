// Chapter 6 playground. Edit this file, then run `npx tsc -p site` to regenerate app.js.

interface Info { embedder: string; reranker: string; llm: boolean; contextModes: string[]; defaultContextMode: string }
interface Hit { source_id: string; title: string; section: string; score: number; text: string; fused_rank?: number; rerank_score?: number; matched_child?: string }
interface PipelineResponse { reranker: string; contextMode: string; queries: string[]; timings: Record<string, number>; fused: Hit[]; final: Hit[] }
interface TransformResponse { multi: { queries: string[]; source: string }; hyde?: { passage: string | null; source: string } }
interface TraceStep { type: string; author: string; name?: string; args?: Record<string, unknown>; response?: { results?: Hit[] }; text?: string }
interface AskResponse { answer?: string; trace?: TraceStep[]; error?: string }

const $ = <T extends HTMLElement = HTMLElement>(id: string): T => document.getElementById(id) as T;
const value = (id: string): string => $<HTMLInputElement>(id).value;
const checked = (id: string): boolean => $<HTMLInputElement>(id).checked;

const escapeHtml = (s: string): string =>
  s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c] ?? c);

async function api<T>(path: string, body: unknown = {}): Promise<T> {
  const res = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error ?? res.statusText);
  return data as T;
}

function setStatus(id: string, msg: string, isError = false): void {
  $(id).textContent = msg;
  $(id).classList.toggle("error", isError);
}

function card(h: Hit, i: number): string {
  return `
    <article class="result">
      <header><span class="title">#${i + 1} ${escapeHtml(h.source_id)}</span><span class="score">${(h.rerank_score ?? h.score).toFixed(4)}</span></header>
      <div class="small">${escapeHtml(h.title)} › ${escapeHtml(h.section)}</div>
      ${h.matched_child ? `<div class="child">matched child: “${escapeHtml(h.matched_child)}”</div>` : ""}
      <div class="snippet">${escapeHtml(h.text.slice(0, 180))}…</div>
    </article>`;
}

const variantLines = (): string[] => value("queries").split("\n").map((l) => l.trim()).filter(Boolean);

async function generate(): Promise<void> {
  setStatus("status", "Generating query variants…");
  try {
    const r = await api<TransformResponse>("/api/transform", { query: value("q"), hyde: checked("useHyde") });
    $<HTMLTextAreaElement>("queries").value = r.multi.queries.join("\n");
    if (r.hyde) $<HTMLTextAreaElement>("hydePassage").value = r.hyde.passage ?? "";
    setStatus("status", `Variants: ${r.multi.source}${r.hyde ? ` · HyDE: ${r.hyde.source}` : ""}`);
  } catch (e) {
    setStatus("status", (e as Error).message, true);
  }
}

async function search(): Promise<void> {
  setStatus("status", "Searching…");
  try {
    const queries = variantLines();
    const r = await api<PipelineResponse>("/api/search", {
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
  } catch (e) {
    setStatus("status", `Could not reach the API (${(e as Error).message}). Run: python server.py`, true);
  }
}

async function compareContext(): Promise<void> {
  try {
    const r = await api<Record<string, Hit[]>>("/api/compare-context", { query: value("cq") });
    $("ccolumns").innerHTML = Object.entries(r).map(([mode, hits]) =>
      `<div><h4>context: ${escapeHtml(mode)}</h4><div class="results">${hits.map(card).join("")}</div></div>`).join("");
  } catch (e) {
    $("ccolumns").innerHTML = `<p class="status error">${escapeHtml((e as Error).message)}</p>`;
  }
}

async function askAgent(): Promise<void> {
  setStatus("ask-status", "Thinking…");
  $("ask").setAttribute("disabled", "");
  try {
    const r = await api<AskResponse>("/api/ask", { question: value("question") });
    if (r.error) { setStatus("ask-status", r.error, true); return; }
    setStatus("ask-status", "");
    const trace = (r.trace ?? []).filter((s) => s.type !== "message").map((s) => s.type === "tool_call"
      ? `<div>🔎 ${escapeHtml(s.name ?? "")}(${escapeHtml(JSON.stringify(s.args))})</div>`
      : `<div>&nbsp;&nbsp;↳ ${escapeHtml((s.response?.results ?? []).map((h) => h.source_id).join(", "))}</div>`).join("");
    $("answer").innerHTML = `<div class="trace">${trace}</div><div class="answer">${escapeHtml(r.answer ?? "")}</div>`;
  } catch (e) {
    setStatus("ask-status", (e as Error).message, true);
  } finally {
    $("ask").removeAttribute("disabled");
  }
}

async function loadInfo(): Promise<void> {
  try {
    const info = await api<Info>("/api/info");
    $("info").innerHTML = `Embedder <code>${escapeHtml(info.embedder)}</code> · reranker <code>${escapeHtml(info.reranker)}</code> · Gemini ${info.llm ? "✅" : "❌ (heuristic fallbacks; LLM context unavailable)"}`;
    $("contextMode").innerHTML = info.contextModes.map((m) => `<option ${m === info.defaultContextMode ? "selected" : ""}>${m}</option>`).join("");
  } catch {
    $("info").textContent = "Server not reachable.";
  }
}

function init(): void {
  const examples = ["can my kid ride on the back and is it legal in the EU?", "how do I clear stored codes", "it's dead, nothing on the screen", "cheapest way to carry groceries and how far can it go"];
  $("examples").innerHTML = examples.map((q) => `<button class="chip" type="button">${escapeHtml(q)}</button>`).join("");
  $("examples").addEventListener("click", (ev) => {
    const t = ev.target as HTMLElement;
    if (!t.classList.contains("chip")) return;
    $<HTMLInputElement>("q").value = t.textContent ?? "";
    $<HTMLTextAreaElement>("queries").value = "";
    $<HTMLTextAreaElement>("hydePassage").value = "";
    void search();
  });
  $("gen").addEventListener("click", () => void generate());
  $("go").addEventListener("click", () => void search());
  $("q").addEventListener("keydown", (ev) => { if ((ev as KeyboardEvent).key === "Enter") void search(); });
  for (const id of ["contextMode", "parentDocs", "rerank"]) $(id).addEventListener("change", () => void search());
  $("cgo").addEventListener("click", () => void compareContext());
  $("ask").addEventListener("click", () => void askAgent());
  void loadInfo().then(() => { void search(); void compareContext(); });
}

init();
