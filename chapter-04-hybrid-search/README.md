# Chapter 4 · Hybrid Search (BM25 + Vectors + RRF)

> Builds on Chapters 1 and 3. Runs lexical and semantic retrieval over the same chunks and fuses them.

## What you'll learn
- Why lexical and semantic retrieval fail on *different* queries
- **Reciprocal Rank Fusion**: formula, why `k = 60`, worked example
- Weighted (convex) fusion with min-max normalisation and `α`
- Hit@3 and MRR on a 12-query scoreboard (a preview of Chapter 7)

Measured on this corpus with the local bge-small embedder:

| Mode | Hit@3 | MRR |
|---|---|---|
| BM25 | 75% | 0.688 |
| Vector | 92% | 0.887 |
| **Hybrid (RRF)** | **100%** | **0.917** |

## Run it
```bash
cd chapter-04-hybrid-search
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env               # key only needed for the agent

python hybrid.py "UN3480"          # BM25 vs vector vs hybrid
python server.py                   # → http://localhost:8004
python ask.py "Can I get a spare battery shipped to Hawaii?"
adk web                            # pick rag_agent
```

## Files
| File | Purpose |
|---|---|
| `hybrid.py` | **New.** `rrf()`, `min_max()`, `HybridRetriever`, `retrieve()` |
| `rag_agent/agent.py` | Now imports `retrieve` from `hybrid.py` |
| `server.py` | Playground API incl. the scoreboard |
| everything else | Same as Chapter 3 |

## Exercises
1. Add a third list to the fusion: BM25 over **titles only**. Does the scoreboard improve?
2. Implement *distribution-based* score fusion (z-score normalisation instead of min-max).
3. Add 5 queries of your own to `SCOREBOARD`, including one where hybrid loses to pure vector.

**Next:** Chapter 5 re-sorts the fused candidates with a reranker.
