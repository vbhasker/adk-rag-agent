# Chapter 7 · Evaluation: Stop Guessing, Start Measuring

> Builds on Chapter 6. Gives every knob from Chapters 1–6 a number.

## What you'll learn
- Building a golden dataset (keyword, semantic, multi-part and **unanswerable** questions)
- Retrieval metrics: Hit@k, Recall@k, Precision@k, MRR, nDCG@k
- Answer metrics with an LLM judge: faithfulness, correctness, refusal accuracy, plus citation validity
- Running the same golden set through **ADK's evaluator** (`adk eval`)

## Run it
```bash
cd chapter-07-evaluation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

python eval_retrieval.py              # leaderboard of 7 configs (no key needed)
python eval_retrieval.py --k 5 --llm  # + multi-query & HyDE (needs key)
python eval_answers.py --limit 5      # agent + LLM judge (needs key)
python make_adk_evalset.py && adk eval rag_agent rag_agent/nimbus.evalset.json --config_file_path rag_agent/test_config.json
python server.py                      # → http://localhost:8007
```

Measured with the local bge-small embedder, no reranker, k = 3 (21 answerable questions):

| Config | Hit@3 | Recall@3 | MRR |
|---|---|---|---|
| BM25 | 1.000 | 0.929 | 0.865 |
| Vector | 0.952 | 0.952 | **0.933** |
| Hybrid (RRF) | **1.000** | **0.976** | 0.921 |

Hybrid never misses completely; vector ranks the first hit slightly better on this small, clean corpus.
Run it with your own setup, since your numbers will differ.

## Files
| File | Purpose |
|---|---|
| `eval/golden.jsonl` | **New.** 24 questions + relevant docs + reference answers |
| `metrics.py` | **New.** Retrieval metrics |
| `eval_retrieval.py` | **New.** Config leaderboard (`CONFIGS` dict: add your own) |
| `eval_answers.py` | **New.** Agent run + LLM-as-judge + citation check |
| `make_adk_evalset.py` | **New.** Golden set → ADK `EvalSet` + `test_config.json` |
| everything else | Same as Chapter 6 |

## Exercises
1. Add 10 questions from "real users" (ask a friend!) and rerun. Does the winner change?
2. Add a config with `candidates=40` and one with chunk size 60. Which knob matters more?
3. Validate the judge: hand-label 10 answers yourself and compare with `eval_answers.py` scores.

**Next:** Chapter 8, agentic RAG.
