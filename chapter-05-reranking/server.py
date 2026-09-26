"""Chapter 5 lesson site + playground API.  Run:  python server.py  ->  http://localhost:8005"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_runner import ask, has_llm_credentials
from hybrid import get_retriever
from pipeline import run_pipeline
from rerank import available_rerankers, get_reranker
from webserver import serve


def api_info(_: dict[str, Any]) -> dict[str, Any]:
    r = get_retriever()
    return {
        "embedder": r.store.embedder_name,
        "chunks": len(r.store.chunks),
        "rerankers": available_rerankers(),
        "defaultReranker": get_reranker().name,
        "llm": has_llm_credentials(),
    }


def api_search(p: dict[str, Any]) -> dict[str, Any]:
    kind = p.get("reranker") or None
    mmr_on = bool(p.get("mmr"))
    min_score = p.get("minScore")
    return run_pipeline(
        str(p.get("query", "")),
        top_k=int(p.get("topK", 5)),
        candidates=int(p.get("candidates", 20)),
        use_rerank=kind != "off",
        reranker_kind=None if kind in (None, "off") else kind,
        mmr_lambda=float(p.get("mmrLambda", 0.7)) if mmr_on else None,
        min_score=float(min_score) if min_score not in (None, "") else None,
        category=p.get("category") or None,
    )


def api_ask(p: dict[str, Any]) -> dict[str, Any]:
    if not has_llm_credentials():
        return {"error": "No Gemini credentials. Add GOOGLE_API_KEY (or Vertex AI settings) to .env and restart."}
    from rag_agent.agent import root_agent

    return ask(root_agent, str(p.get("question", "")))


if __name__ == "__main__":
    get_retriever()
    get_reranker()
    serve(
        {"/api/info": api_info, "/api/search": api_search, "/api/ask": api_ask},
        site_dir=Path(__file__).parent / "site",
        port=8005,
    )
