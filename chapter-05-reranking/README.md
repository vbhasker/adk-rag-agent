# Chapter 5 · Reranking, Diversity (MMR) & Filters

> Builds on Chapter 4. Adds a second retrieval stage: hybrid top 20 → reranker → top 5.

## What you'll learn
- Two-stage retrieval: recall first (cheap), precision second (smart)
- Bi-encoders vs **cross-encoders**; LLM-as-reranker with Gemini structured output
- **MMR** for diverse results; relevance **thresholds** that let the agent say "I don't know"
- Metadata filters as a precision *and* permissions tool

## Run it
```bash
cd chapter-05-reranking
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

python pipeline.py "what happens if the motor gets too hot"   # before/after reranking
python pipeline.py "warranty" --mmr 0.5                       # diversity
python server.py                                              # → http://localhost:8005
python ask.py "My display shows E-07 after a long climb. What should I do?"
adk web
```
The cross-encoder (`Xenova/ms-marco-MiniLM-L-6-v2`, ~80 MB) downloads on first use. Without it, the
pipeline falls back to the LLM reranker (Gemini, or OpenAI with `LLM_PROVIDER=openai`) if a key is set, else no reranking.

## Files
| File | Purpose |
|---|---|
| `rerank.py` | **New.** Cross-encoder, LLM (Gemini/OpenAI) and no-op rerankers |
| `mmr.py` | **New.** Maximal Marginal Relevance |
| `pipeline.py` | **New.** hybrid → rerank → threshold → MMR → top k, with timings; `retrieve()` for the agent |
| `llm.py` | **New.** Gemini helpers (`generate_json` with Pydantic schemas) |
| `rag_agent/agent.py` | Uses `pipeline.retrieve` |
| everything else | Same as Chapter 4 |

## Exercises
1. Add the Vertex AI ranking API (Discovery Engine `RankService`) as a fourth reranker.
2. Rerank with the cross-encoder, then let Gemini rerank only the top 5. Is the cascade worth it?
3. Find a `RERANK_MIN_SCORE` that removes all chunks for 5 off-topic questions but keeps the answer for 5 on-topic ones.

**Next:** Chapter 6 fixes bad *queries* and context-poor *chunks*.
