# Chapter 3 · Chunking, Vector Store & Your First ADK RAG Agent

> Builds on Chapter 2. First complete RAG loop: chunk → embed → store → retrieve → Gemini answers with citations.

## What you'll learn
- Why chunking matters; fixed vs sentence vs structure-aware (Markdown) chunking; overlap
- Adding heading context to chunk embeddings
- What a vector store does: exact vs ANN search (HNSW/IVF/ScaNN), metadata pre-filtering, persistence
- Retrieval as an **ADK tool**; grounding + citation instructions

## Run it
```bash
cd chapter-03-chunking-vector-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # add GOOGLE_API_KEY for the agent

python ingest.py --query "is a speed chip covered by warranty?"   # retrieval only, no key needed
python server.py                  # lesson + playground → http://localhost:8003
python ask.py "How should I store my battery over winter?"        # the agent (needs key)
adk web                           # ADK dev UI → pick "rag_agent"
```

## Files
| File | Purpose |
|---|---|
| `chunking.py` | `fixed`, `sentence`, `markdown` (structure-aware, recursive) chunkers |
| `vector_store.py` | NumPy vector store: exact top-k, metadata filter, save/load with fingerprint |
| `ingest.py` | `build_index()` (load → chunk → embed → store) and `retrieve()` |
| `rag_agent/agent.py` | ADK `root_agent` with a `search_knowledge_base` tool |
| `agent_runner.py`, `ask.py` | Run the agent from Python and print its tool-call trace |
| `server.py` | Playground API: chunk visualizer, search, ask |
| `embeddings.py`, `bm25.py`, `corpus.py`, … | Carried over from Chapters 1–2 |

The index is saved in `.index/` and rebuilds automatically when docs, chunking or the embedder change.

## Exercises
1. Add a `page_or_section_number` metadata field and make the agent cite it.
2. Implement **semantic chunking**: split a section where cosine similarity between consecutive sentences drops below a threshold.
3. Change the agent into a fixed pipeline: a `before_model_callback` that always retrieves and injects context. Compare behaviour.

**Next:** Chapter 4 combines BM25 and vectors with hybrid search.
