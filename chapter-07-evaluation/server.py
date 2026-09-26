"""Chapter 7 lesson site + playground API.  Run:  python server.py  ->  http://localhost:8007"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import llm
from eval_retrieval import CONFIGS, LLM_CONFIGS, RESULTS_DIR, evaluate, load_golden
from hybrid import get_retriever
from rerank import get_reranker
from webserver import serve

_CACHE: dict[tuple[int, bool], dict[str, Any]] = {}


def api_info(_: dict[str, Any]) -> dict[str, Any]:
    return {"embedder": get_retriever().store.embedder_name, "reranker": get_reranker().name, "llm": llm.has_llm_credentials()}


def api_golden(_: dict[str, Any]) -> list[dict[str, Any]]:
    return load_golden()


def api_eval_retrieval(p: dict[str, Any]) -> dict[str, Any]:
    k, use_llm = int(p.get("k", 3)), bool(p.get("llm")) and llm.has_llm_credentials()
    if (k, use_llm) not in _CACHE:
        _CACHE[(k, use_llm)] = evaluate({**CONFIGS, **(LLM_CONFIGS if use_llm else {})}, k)
    return _CACHE[(k, use_llm)]


def api_answers(p: dict[str, Any]) -> dict[str, Any]:
    """Return saved answer-eval results; with run=true (and a key), run the evaluation first."""
    path = RESULTS_DIR / "answers.json"
    if p.get("run"):
        if not llm.has_llm_credentials():
            return {"error": "No LLM credentials. Add GOOGLE_API_KEY (or Vertex AI settings) to .env, or set LLM_PROVIDER=openai with OPENAI_API_KEY, then restart."}
        from eval_answers import evaluate_question, summarise
        from rag_agent.agent import root_agent

        golden = load_golden()[: int(p.get("limit", 24))]
        rows = [evaluate_question(root_agent, g) for g in golden]
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"model": llm.model_name(), "summary": summarise(rows), "rows": rows}, indent=1))
    if not path.exists():
        return {"error": "No answer-eval results yet. Run `python eval_answers.py` (needs a key) or press Run."}
    return json.loads(path.read_text())


if __name__ == "__main__":
    api_eval_retrieval({"k": 3})  # warm the cache (builds all indexes)
    serve(
        {"/api/info": api_info, "/api/golden": api_golden, "/api/eval-retrieval": api_eval_retrieval, "/api/answers": api_answers},
        site_dir=Path(__file__).parent / "site",
        port=8007,
    )
