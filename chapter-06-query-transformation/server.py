"""Chapter 6 lesson site + playground API.  Run:  python server.py  ->  http://localhost:8006"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_runner import ask, has_llm_credentials
from hybrid import get_retriever
from ingest import CONTEXTS_FILE, default_context_mode, get_store
from pipeline import run_pipeline
from query_transform import hyde, multi_query, rewrite
from rerank import get_reranker
from webserver import serve


def context_modes() -> list[str]:
    return ["none", "header"] + (["llm"] if CONTEXTS_FILE.exists() else [])


def api_info(_: dict[str, Any]) -> dict[str, Any]:
    return {
        "embedder": get_store().embedder_name,
        "reranker": get_reranker().name,
        "llm": has_llm_credentials(),
        "contextModes": context_modes(),
        "defaultContextMode": default_context_mode(),
    }


def api_transform(p: dict[str, Any]) -> dict[str, Any]:
    query = str(p.get("query", ""))
    history = [h for h in p.get("history", []) if h]
    out: dict[str, Any] = {"multi": multi_query(query)}
    if p.get("hyde"):
        out["hyde"] = hyde(query)
    if history:
        out["rewrite"] = rewrite(query, history)
    return out


def api_search(p: dict[str, Any]) -> dict[str, Any]:
    query = str(p.get("query", ""))
    queries = [q.strip() for q in p.get("queries", []) if q.strip()] or [query]
    return run_pipeline(
        query,
        top_k=int(p.get("topK", 5)),
        queries=queries,
        hyde_passage=p.get("hydePassage") or None,
        context_mode=p.get("contextMode") or None,
        parent_docs=bool(p.get("parentDocs")),
        use_rerank=bool(p.get("rerank", True)),
    )


def api_compare_context(p: dict[str, Any]) -> dict[str, Any]:
    """Same query, same pipeline, different chunk text: none vs header vs llm context."""
    query = str(p.get("query", ""))
    return {mode: run_pipeline(query, top_k=4, context_mode=mode)["final"] for mode in context_modes()}


def api_ask(p: dict[str, Any]) -> dict[str, Any]:
    if not has_llm_credentials():
        return {"error": "No LLM credentials. Add GOOGLE_API_KEY (or Vertex AI settings) to .env, or set LLM_PROVIDER=openai with OPENAI_API_KEY, then restart."}
    from rag_agent.agent import root_agent

    return ask(root_agent, str(p.get("question", "")))


if __name__ == "__main__":
    for mode in context_modes():
        get_retriever(mode)  # build every index up front
    get_reranker()
    serve(
        {
            "/api/info": api_info,
            "/api/transform": api_transform,
            "/api/search": api_search,
            "/api/compare-context": api_compare_context,
            "/api/ask": api_ask,
        },
        site_dir=Path(__file__).parent / "site",
        port=8006,
    )
