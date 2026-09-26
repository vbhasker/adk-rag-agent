# Chapter 6 · Query Transformation & Contextual Retrieval

> Builds on Chapter 5. Better queries in (rewrite, multi-query, decomposition, HyDE), richer chunks
> waiting (contextual headers, LLM-generated context, parent-document retrieval).

## What you'll learn
- Query rewriting for follow-ups; multi-query + decomposition fused with RRF; HyDE; step-back
- Why a tool-calling agent gets query rewriting "for free", and how a `queries: list[str]` tool
  parameter gives you multi-query without extra LLM calls
- Contextual retrieval: `none` vs `header` vs `llm` chunk context
- Parent-document (small-to-big) retrieval

## Run it
```bash
cd chapter-06-query-transformation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

python query_transform.py "can my kid ride on the back and is it legal in the EU?"
python pipeline.py "how do I clear stored codes" --parent --context none
python contextualize.py            # optional, needs a key: LLM context per chunk (cached)
python server.py                   # → http://localhost:8006
python ask.py "Can my 8-year-old ride on the back, and is a throttle legal in the EU?"
adk web
```
Without a Gemini key, query transforms fall back to clearly-labelled heuristics, and you can type
query variants / a HyDE passage by hand in the playground.

## Files
| File | Purpose |
|---|---|
| `query_transform.py` | **New.** `rewrite`, `multi_query`, `hyde` |
| `contextualize.py` | **New.** Generates and caches LLM context per chunk |
| `parent.py` | **New.** Child chunks + small-to-big mapping |
| `chunking.py` | `Chunk.context`, `text_for_embedding(mode)` |
| `ingest.py` | Context modes, one index per mode |
| `hybrid.py` | `vector_text` for HyDE; one retriever per mode |
| `pipeline.py` | Multi-list RRF, parent swap, rerank vs original question |
| `rag_agent/agent.py` | Tool takes `queries: list[str]` |

## Exercises
1. Implement **step-back** prompting as a fourth transform and add it to the playground.
2. Run `contextualize.py`, then compare `header` vs `llm` context in the playground. Find a query where LLM context wins.
3. Try `sentence-window` retrieval: return the matched child plus its neighbours instead of the whole parent.

**Next:** Chapter 7 measures all of this properly.
