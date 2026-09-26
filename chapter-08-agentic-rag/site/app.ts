// Chapter 8 playground. Edit this file, then run `npx -p typescript tsc -p site` to regenerate app.js.

interface Info { embedder: string; reranker: string; llm: boolean }
interface Hit { source_id: string; title?: string; section?: string; text?: string }
interface TraceStep { type: "tool_call" | "tool_result" | "message"; author: string; name?: string; args?: Record<string, unknown>; response?: { results?: Hit[]; doc_id?: string; documents?: unknown[]; error?: string }; text?: string; t: number }
interface AskResponse { answer?: string; trace?: TraceStep[]; state?: Record<string, string>; seconds?: number; error?: string }

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

// ---------- Tools explorer ----------
const TOOL_HINTS: Record<string, [string, string]> = {
  search_knowledge_base: ["battery warranty cycles | battery lifespan cycles", "Queries separated by | (the tool takes a list)."],
  get_document: ["warranty", "A doc id, e.g. warranty, battery-care, error-codes."],
  list_documents: ["", "No arguments."],
};

function toolArgs(): Record<string, unknown> {
  const tool = value("tool");
  const raw = value("tool-arg").trim();
  if (tool === "search_knowledge_base") return { queries: raw.split("|").map((q) => q.trim()).filter(Boolean) };
  if (tool === "get_document") return { doc_id: raw };
  return {};
}

async function callTool(): Promise<void> {
  try {
    const out = await api<unknown>("/api/tool", { name: value("tool"), args: toolArgs() });
    $("tool-out").textContent = JSON.stringify(out, null, 2);
  } catch (e) {
    $("tool-out").textContent = `Error: ${(e as Error).message} (is python server.py running?)`;
  }
}

// ---------- Agents ----------
function describe(step: TraceStep): string {
  if (step.type === "tool_call") return `🔧 ${step.name}(${JSON.stringify(step.args)})`;
  if (step.type === "tool_result") {
    const r = step.response ?? {};
    if (r.results) return `↳ ${r.results.map((h) => h.source_id).join(", ") || "no results"}`;
    if (r.doc_id) return `↳ read full document "${r.doc_id}"`;
    if (r.documents) return `↳ ${r.documents.length} documents listed`;
    if (step.name === "exit_loop") return "↳ ✅ draft approved, loop exits";
    return `↳ ${JSON.stringify(r).slice(0, 120)}`;
  }
  return `💬 ${step.text ?? ""}`;
}

function renderRun(r: AskResponse): string {
  const steps = (r.trace ?? []).map((s) => `
    <div class="step"><span class="t">${s.t.toFixed(1)}s</span><span class="who ${escapeHtml(s.author)}">${escapeHtml(s.author)}</span><span class="what">${escapeHtml(describe(s))}</span></div>`).join("");
  const toolCalls = (r.trace ?? []).filter((s) => s.type === "tool_call" && s.name !== "exit_loop").length;
  const rounds = (r.trace ?? []).filter((s) => s.author === "writer" && s.type === "message").length;
  const meta = `${r.seconds}s · ${toolCalls} retrieval tool call${toolCalls === 1 ? "" : "s"}${rounds ? ` · ${rounds} writer round${rounds === 1 ? "" : "s"}` : ""}`;
  return `<div class="answer">${escapeHtml(r.answer ?? "")}</div><p class="small muted">${meta}</p><details><summary>Trace</summary><div class="timeline">${steps}</div></details>`;
}

async function askAgent(kind: "simple" | "agentic"): Promise<void> {
  const statusId = `${kind}-status`;
  setStatus(statusId, kind === "agentic" ? "Researching, writing, fact-checking…" : "Thinking…");
  $(`ask-${kind}`).setAttribute("disabled", "");
  try {
    const r = await api<AskResponse>("/api/ask", { question: value("question"), agent: kind });
    if (r.error) { setStatus(statusId, r.error, true); return; }
    setStatus(statusId, "");
    $(`${kind}-out`).innerHTML = renderRun(r);
  } catch (e) {
    setStatus(statusId, (e as Error).message, true);
  } finally {
    $(`ask-${kind}`).removeAttribute("disabled");
  }
}

async function loadInfo(): Promise<void> {
  try {
    const info = await api<Info>("/api/info");
    $("info").innerHTML = `Embedder <code>${escapeHtml(info.embedder)}</code> · reranker <code>${escapeHtml(info.reranker)}</code> · LLM ${info.llm ? "✅" : "❌ (tools explorer works; agents need a key)"}`;
  } catch {
    $("info").textContent = "Server not reachable.";
  }
}

function init(): void {
  $("tool").addEventListener("change", () => {
    const [arg, hint] = TOOL_HINTS[value("tool")];
    $<HTMLInputElement>("tool-arg").value = arg;
    $("tool-hint").textContent = hint;
    void callTool();
  });
  $("tool-run").addEventListener("click", () => void callTool());
  const examples = ["Compare the battery warranty with the battery's expected lifespan.", "How much does the Nimbus Trail cost?", "Can my kid ride on the back, and is a throttle legal in the EU?", "My display shows E07 after a long climb. Is it covered by warranty if the motor is damaged?", "hi there!"];
  $("examples").innerHTML = examples.map((q) => `<button class="chip" type="button">${escapeHtml(q)}</button>`).join("");
  $("examples").addEventListener("click", (ev) => {
    const t = ev.target as HTMLElement;
    if (t.classList.contains("chip")) $<HTMLInputElement>("question").value = t.textContent ?? "";
  });
  $("ask-simple").addEventListener("click", () => void askAgent("simple"));
  $("ask-agentic").addEventListener("click", () => void askAgent("agentic"));
  void loadInfo();
  void callTool();
}

init();
