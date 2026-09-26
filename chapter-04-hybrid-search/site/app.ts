// Chapter 4 playground. Edit this file, then run `npx -p typescript tsc -p site` to regenerate app.js.

interface Info { embedder: string; chunks: number; llm: boolean }
interface Hit { source_id: string; title: string; section: string; score: number; bm25_rank: number | null; vector_rank: number | null; text: string }
type Mode = "bm25" | "vector" | "hybrid";
type SearchResponse = Record<Mode, Hit[]>;
interface ScoreRow { query: string; expected: string; bm25: number | null; vector: number | null; hybrid: number | null }
interface Scoreboard { rows: ScoreRow[]; summary: Record<Mode, { hitAt3: number; mrr: number }> }
interface TraceStep { type: string; author: string; name?: string; args?: Record<string, unknown>; response?: { results?: Hit[] }; text?: string }
interface AskResponse { answer?: string; trace?: TraceStep[]; error?: string }

const MODES: Mode[] = ["bm25", "vector", "hybrid"];
const LABELS: Record<Mode, string> = { bm25: "BM25 only", vector: "Vector only", hybrid: "Hybrid (fused)" };

const $ = <T extends HTMLElement = HTMLElement>(id: string): T => document.getElementById(id) as T;
const value = (id: string): string => $<HTMLInputElement>(id).value;

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

const fusionSettings = () => ({ fusion: value("fusion"), alpha: Number(value("alpha")), rrfK: Number(value("rrfK")) });

const rankBadge = (label: string, rank: number | null) =>
  `<span class="rank${rank ? "" : " none"}">${label} ${rank ? `#${rank}` : "–"}</span>`;

function column(mode: Mode, hits: Hit[]): string {
  const cards = hits.map((h, i) => `
    <article class="result">
      <header><span class="title">#${i + 1} ${escapeHtml(h.source_id)}</span><span class="score">${h.score.toFixed(4)}</span></header>
      <div class="small">${escapeHtml(h.section || h.title)}</div>
      ${mode === "hybrid" ? `<div style="margin-top:6px">${rankBadge("bm25", h.bm25_rank)}${rankBadge("vec", h.vector_rank)}</div>` : ""}
      <div class="snippet">${escapeHtml(h.text.slice(0, 140))}…</div>
    </article>`).join("");
  return `<div><h4>${LABELS[mode]}</h4><div class="results">${cards || '<p class="muted">No results.</p>'}</div></div>`;
}

async function search(): Promise<void> {
  setStatus("status", "Searching both retrievers…");
  try {
    const r = await api<SearchResponse>("/api/search", { query: value("q"), topK: 5, ...fusionSettings() });
    $("columns").innerHTML = MODES.map((m) => column(m, r[m])).join("");
    setStatus("status", "");
  } catch (e) {
    setStatus("status", `Could not reach the API (${(e as Error).message}). Run: python server.py`, true);
  }
}

async function scoreboard(): Promise<void> {
  try {
    const r = await api<Scoreboard>("/api/scoreboard", fusionSettings());
    const cell = (rank: number | null) => rank && rank <= 3 ? `<span class="ok">#${rank}</span>` : `<span class="miss">${rank ? `#${rank}` : "miss"}</span>`;
    const summary = MODES.map((m) => `<th>${LABELS[m]}<br /><span style="text-transform:none">hit@3 ${(r.summary[m].hitAt3 * 100).toFixed(0)}% · MRR ${r.summary[m].mrr.toFixed(3)}</span></th>`).join("");
    const rows = r.rows.map((row) => `<tr><td>${escapeHtml(row.query)}<br /><span class="small muted">→ ${escapeHtml(row.expected)}</span></td>${MODES.map((m) => `<td>${cell(row[m])}</td>`).join("")}</tr>`).join("");
    $("scoreboard").innerHTML = `<div class="table-wrap"><table><tr><th>Query → answer doc</th>${summary}</tr>${rows}</table></div>`;
  } catch {
    $("scoreboard").innerHTML = `<p class="muted">Start the server to compute the scoreboard.</p>`;
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
    $("info").innerHTML = `Embedder <code>${escapeHtml(info.embedder)}</code> · ${info.chunks} chunks · LLM ${info.llm ? "✅" : "❌ (add a key for the agent)"}`;
  } catch {
    $("info").textContent = "Server not reachable.";
  }
}

function init(): void {
  const examples = ["UN3480", "E-12", "E07", "my bike won't turn on", "what happens if the motor gets too hot", "can I send it back if I don't like it", "firmware 4.2.1"];
  $("examples").innerHTML = examples.map((q) => `<button class="chip" type="button">${escapeHtml(q)}</button>`).join("");
  $("examples").addEventListener("click", (ev) => {
    const t = ev.target as HTMLElement;
    if (t.classList.contains("chip")) { $<HTMLInputElement>("q").value = t.textContent ?? ""; void search(); }
  });
  for (const id of ["rrfK", "alpha"]) {
    $(id).addEventListener("change", () => { void search(); void scoreboard(); });
    $(id).addEventListener("input", () => { $(`${id}v`).textContent = value(id); });
  }
  $("fusion").addEventListener("change", () => { void search(); void scoreboard(); });
  $("go").addEventListener("click", () => void search());
  $("q").addEventListener("keydown", (ev) => { if ((ev as KeyboardEvent).key === "Enter") void search(); });
  $("ask").addEventListener("click", () => void askAgent());
  void loadInfo();
  void search();
  void scoreboard();
}

init();
