# 🚲 RAG Weekend: from BM25 to Agentic RAG with Google ADK

An 8-chapter, hands-on course that takes you from "what is keyword search?" to a fact-checked multi-agent RAG
system in Google ADK. Every chapter is a **standalone, runnable project** with its own **lesson website +
live playground**, and each one builds on the previous.

All chapters use the same knowledge base: **Nimbus Bikes**, a fictional e-bike company with 12 docs (specs,
warranty, battery care, error codes, returns…). It's small enough to read in 10 minutes, and it's
deliberately full of both exact codes (`E-07`, `UN3480`) *and* fuzzy human questions ("my bike won't turn on"),
so you can watch every technique win and lose.

> Open `index.html` in your browser for the course hub, or start the first lesson:
> `cd chapter-01-keyword-search-bm25 && python server.py`

---

## The map

| # | Chapter | You'll learn | Needs a key? | Port |
|---|---|---|---|---|
| 1 | [Keyword Search & BM25](chapter-01-keyword-search-bm25) | Tokenization, inverted index, TF-IDF, BM25 (`k1`, `b`), vocabulary mismatch | No (stdlib only!) | 8001 |
| 2 | [Embeddings & Similarity](chapter-02-embeddings-similarity) | Dense vectors, cosine/dot/Euclidean, asymmetric & Matryoshka embeddings, semantic search | No (local model) | 8002 |
| 3 | [Chunking, Vector Store & first ADK agent](chapter-03-chunking-vector-rag) | Fixed/sentence/structure-aware chunking, vector store, ANN, retrieval as an ADK tool, citations | Agent only | 8003 |
| 4 | [Hybrid Search](chapter-04-hybrid-search) | BM25 + vectors, Reciprocal Rank Fusion, weighted fusion, Hit@k & MRR | Agent only | 8004 |
| 5 | [Reranking, MMR & Filters](chapter-05-reranking) | Two-stage retrieval, cross-encoders, LLM reranking, MMR diversity, thresholds | Agent only | 8005 |
| 6 | [Query Transformation & Contextual Retrieval](chapter-06-query-transformation) | Rewrite, multi-query, decomposition, HyDE, contextual chunks, parent-document retrieval | Partly | 8006 |
| 7 | [Evaluation](chapter-07-evaluation) | Golden datasets, Recall/MRR/nDCG, LLM-as-judge faithfulness, `adk eval` | Answer eval only | 8007 |
| 8 | [Agentic RAG](chapter-08-agentic-rag) | Researcher → writer ⟲ fact-checker with `SequentialAgent`/`LoopAgent`, 2026 landscape, project checklist | Agents only | 8008 |

```
Ch1 BM25 ─┐
          ├─► Ch4 Hybrid ─► Ch5 Rerank ─► Ch6 Query/Chunk transforms ─► Ch7 Eval ─► Ch8 Agentic
Ch2 Vectors ─► Ch3 Chunks + Vector store + ADK agent ─┘
```

## 🗓️ Your weekend game plan (coach's orders)

| When | Chapters | Goal |
|---|---|---|
| **Friday evening** (~2 h) | 1–2 | Understand *how search ranks things*: keywords vs meaning. No keys, no cloud. |
| **Saturday morning** (~3 h) | 3–4 | First real RAG agent in ADK, then make retrieval robust with hybrid search. |
| **Saturday afternoon** (~3 h) | 5–6 | Precision (reranking) and fixing messy queries and orphan chunks. |
| **Sunday morning** (~2 h) | 7 | Measure everything. This is the chapter that makes you dangerous at work. |
| **Sunday afternoon** (~2 h) | 8 | Agentic RAG, the 2026 landscape, and a plan for your own project. |

For each chapter: **read the lesson page → run the playground → do the "Try this" list → skim "Walk the code" → one exercise from the README.**

## Setup (once)

- Python **3.10+** (tested with 3.11), a browser. Node is only needed if you want to edit the playground TypeScript.
- Each chapter from 2 onward has its own `requirements.txt` (standalone on purpose):

```bash
cd chapter-0X-...
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # then edit
python server.py                                       # lesson + playground
```

### Gemini access (needed from Chapter 3 for the agent, unless you use OpenAI below)

Pick one and put it in each chapter's `.env`:

- **Gemini API key** (fastest): get one at <https://aistudio.google.com/apikey>, then uncomment `GOOGLE_API_KEY=...` (keep `GOOGLE_GENAI_USE_VERTEXAI=FALSE`).
- **Vertex AI**: `gcloud auth application-default login`, then set `GOOGLE_GENAI_USE_VERTEXAI=TRUE`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`.

The agent model defaults to `GEMINI_MODEL=gemini-2.5-flash`. If Google has retired it by the time you run this, set the current Flash model name.

### Using OpenAI instead of (or alongside) Gemini

Every chapter's `.env.example` has an OpenAI block, commented out until you add your key:

```ini
LLM_PROVIDER=openai                 # agents + helper LLM calls (rewrite, HyDE, LLM reranker, judge)
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5-mini             # any chat model your key can use
EMBEDDINGS_PROVIDER=openai          # optional: also use OpenAI embeddings
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

- **Agents (Ch 3–8)** stay pure ADK. `llm_config.agent_model()` returns ADK's `LiteLlm("openai/<OPENAI_MODEL>")`
  instead of a Gemini model name, so tools, instructions and workflows are unchanged. Needs `litellm` (in `requirements.txt`).
- **Helper calls (Ch 5–8)** in `llm.py` use the OpenAI Responses API with structured output (same Pydantic schemas).
- **Embeddings (Ch 2–8)**: `EMBEDDINGS_PROVIDER=openai`, or leave `auto` with only an OpenAI key set. Switching
  embedders rebuilds the index automatically (it's fingerprinted by model).
- You can mix providers, e.g. Gemini for the agent and OpenAI embeddings, or the other way round.
- `adk eval` (Ch 7) uses ADK's own LLM judges, which default to Gemini.

### Embeddings, rerankers, and running offline

`EMBEDDINGS_PROVIDER=auto` uses **Gemini embeddings** (`gemini-embedding-001`) when a key is present, otherwise
the local **`BAAI/bge-small-en-v1.5`** via fastembed (downloads ~70 MB once). If neither works (no network), a
clearly-labelled **toy hashing embedder** keeps everything running. It's *not* semantic, and the playground tells you so.
The Chapter 5+ cross-encoder (`Xenova/ms-marco-MiniLM-L-6-v2`) falls back to the Gemini reranker, then to "none".

**Heads-up:** results differ between embedders. The numbers quoted in the lessons were measured with the local
`bge-small` model and no reranker. Rerun them with your setup; that's literally Chapter 7's lesson.

## How the code grows

Each chapter copies the previous chapter's modules and adds a few new ones, so you can `diff` two chapters to see exactly what changed:

```bash
diff chapter-03-chunking-vector-rag/rag_agent/agent.py chapter-04-hybrid-search/rag_agent/agent.py
```

| Module | Born in | Job |
|---|---|---|
| `corpus.py`, `bm25.py`, `webserver.py` | Ch 1 | Load docs; BM25 from scratch; tiny stdlib web server |
| `embeddings.py`, `similarity.py` | Ch 2 | Gemini/local/toy embedders with cache; similarity math |
| `chunking.py`, `vector_store.py`, `ingest.py`, `rag_agent/`, `agent_runner.py` | Ch 3 | Chunk → embed → store; the ADK agent |
| `hybrid.py` | Ch 4 | BM25 + vector + RRF |
| `rerank.py`, `mmr.py`, `pipeline.py`, `llm.py` | Ch 5 | Two-stage retrieval |
| `query_transform.py`, `contextualize.py`, `parent.py` | Ch 6 | Query and chunk upgrades |
| `metrics.py`, `eval_*.py`, `eval/golden.jsonl`, `make_adk_evalset.py` | Ch 7 | Measurement |
| `tools.py`, multi-agent `rag_agent/`, `simple_agent/` | Ch 8 | Agentic RAG |

Front end: each `site/` has `index.html` (lesson), `style.css`, and the playground written in **TypeScript**
(`app.ts`), compiled to `app.js` so nothing needs building. After editing: `npx -p typescript tsc -p site`.

## Other approaches worth knowing (and when to reach for them)

- **Fully managed:** Vertex AI RAG Engine or Vertex AI Search, plugged into ADK via `VertexAiRagRetrieval` / `VertexAiSearchTool`. Fastest path to production when you don't need custom retrieval logic.
- **Postgres-only stack:** pgvector (or AlloyDB) + full-text/BM25 extension, with hybrid search in SQL. One database, strong ops story.
- **Long-context first:** for a small, stable corpus (like Nimbus!), putting everything in Gemini's context with context caching can beat RAG on simplicity. Measure cost and latency.
- **GraphRAG:** when questions are about relationships across many documents.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Playground says "Could not reach the API" | You opened the HTML file directly. Run `python server.py` and use `http://localhost:800X`. |
| "Toy hashing embedder active" | fastembed couldn't download the model (offline/proxy). Set a Gemini key, or pre-download and set `LOCAL_EMBEDDING_PATH`. |
| Agent answers "No LLM credentials" | Add `GOOGLE_API_KEY` (or Vertex settings), or `LLM_PROVIDER=openai` + `OPENAI_API_KEY`, to that chapter's `.env` and restart. |
| `LLM_PROVIDER=openai` fails with an import error | `pip install -r requirements.txt` (needs `openai` and `litellm`). |
| `404 model not found` | Set `GEMINI_MODEL` (or `OPENAI_MODEL`) to a currently available model. |
| Stale results after editing docs | Delete the chapter's `.index/` and `.cache/` folders (they also auto-rebuild when content changes). |
