"""Chapter 8 lesson site + playground API.  Run:  python server.py  ->  http://localhost:8008"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_runner import ask, has_llm_credentials
from ask import load_agent
from hybrid import get_retriever
from rerank import get_reranker
from tools import get_document, list_documents, search_knowledge_base
from webserver import serve

TOOLS = {"list_documents": list_documents, "search_knowledge_base": search_knowledge_base, "get_document": get_document}


def api_info(_: dict[str, Any]) -> dict[str, Any]:
    return {"embedder": get_retriever().store.embedder_name, "reranker": get_reranker().name, "llm": has_llm_credentials()}


def api_tool(p: dict[str, Any]) -> Any:
    """Call a tool directly, exactly as the agent would (works without a key)."""
    return TOOLS[p["name"]](**p.get("args", {}))


def api_ask(p: dict[str, Any]) -> dict[str, Any]:
    if not has_llm_credentials():
        return {"error": "No LLM credentials. Add GOOGLE_API_KEY (or Vertex AI settings) to .env, or set LLM_PROVIDER=openai with OPENAI_API_KEY, then restart."}
    agent, answer_key = load_agent(p.get("agent", "agentic"))
    return ask(agent, str(p.get("question", "")), answer_key)


if __name__ == "__main__":
    get_retriever()
    get_reranker()
    serve({"/api/info": api_info, "/api/tool": api_tool, "/api/ask": api_ask}, site_dir=Path(__file__).parent / "site", port=8008)
