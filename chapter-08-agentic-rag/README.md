# Chapter 8 · Agentic RAG with Google ADK (+ the 2026 landscape)

> Builds on everything. A research team of ADK agents: **researcher** (tools) → **writer** ⟲ **fact_checker**.

## What you'll learn
- Pipeline RAG vs tool RAG vs agentic RAG, and when each is worth the cost
- Multi-agent workflows with `SequentialAgent`, `LoopAgent`, `output_key` state and `{key?}` templating
- Corrective / self-reflective RAG: explicit GAPS, fact-checking, bounded revision loops with `exit_loop`
- The 2026 landscape: Vertex AI RAG Engine & Search tools, GraphRAG, multimodal/late-interaction retrieval,
  long context + caching, MCP/A2A, memory, security (prompt injection via documents, ACLs), deployment

## Run it
```bash
cd chapter-08-agentic-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # add GOOGLE_API_KEY

python server.py                  # → http://localhost:8008 (tools explorer works without a key)
python ask.py "Compare the battery warranty with the battery's expected lifespan."
python ask.py --agent simple "Compare the battery warranty with the battery's expected lifespan."
python eval_answers.py --agent agentic && python eval_answers.py --agent simple   # the real comparison
adk web                           # pick rag_agent (team) or simple_agent (baseline)
```

## Files
| File | Purpose |
|---|---|
| `tools.py` | **New.** `list_documents`, `search_knowledge_base`, `get_document` |
| `rag_agent/agent.py` | **New.** researcher → LoopAgent(writer, fact_checker) |
| `simple_agent/agent.py` | Chapter 6's single agent, as the baseline |
| `agent_runner.py` | Returns trace **and** final session state |
| `ask.py`, `eval_answers.py` | `--agent simple\|agentic` |
| everything else | The full Chapter 1–7 retrieval + eval stack |

## Exercises
1. Add a **router**: a small `LlmAgent` that sends easy questions to `simple_agent` and hard ones to the team.
2. Give the fact checker the search tool so it can verify claims against the knowledge base, not just the notes.
3. Swap `search_knowledge_base` for ADK's `VertexAiSearchTool` or `VertexAiRagRetrieval` on your own corpus.
4. Expose `tools.py` as an MCP server and connect it with `McpToolset`.

🏁 **Course complete.** See the root README for the full map and the project checklist.
