"""Chapter 3 lesson site + playground API.  Run:  python server.py  ->  http://localhost:8003"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_runner import ask, has_llm_credentials
from chunking import chunk_document
from corpus import load_documents
from embeddings import get_embedder
from ingest import build_index, chunk_to_dict, get_store
from webserver import serve

DOCS = {d.id: d for d in load_documents()}


def api_info(_: dict[str, Any]) -> dict[str, Any]:
    store = get_store()
    return {
        "embedder": store.embedder_name,
        "chunks": len(store.chunks),
        "llm": has_llm_credentials(),
        "docs": [{"id": d.id, "title": d.title} for d in DOCS.values()],
    }


def api_chunk(p: dict[str, Any]) -> dict[str, Any]:
    doc = DOCS[p.get("docId", "warranty")]
    chunks = chunk_document(doc, p.get("strategy", "markdown"), int(p.get("size", 120)), int(p.get("overlap", 20)))
    return {"chunks": [{"id": c.id, "section": c.section, "text": c.text, "words": len(c.text.split())} for c in chunks]}


def api_search(p: dict[str, Any]) -> dict[str, Any]:
    store = build_index(p.get("strategy", "markdown"), int(p.get("size", 120)), int(p.get("overlap", 20)))
    query_vector = get_embedder().embed_query(str(p.get("query", "")))
    category = p.get("category") or None
    hits = store.search(query_vector, int(p.get("topK", 4)), where={"category": category} if category else None)
    return {"chunkCount": len(store.chunks), "results": [chunk_to_dict(c, s) for c, s in hits]}


def api_ask(p: dict[str, Any]) -> dict[str, Any]:
    if not has_llm_credentials():
        return {"error": "No Gemini credentials. Add GOOGLE_API_KEY (or Vertex AI settings) to .env and restart."}
    from rag_agent.agent import root_agent

    return ask(root_agent, str(p.get("question", "")))


if __name__ == "__main__":
    get_store()  # build the index up front so the first search is fast
    serve(
        {"/api/info": api_info, "/api/chunk": api_chunk, "/api/search": api_search, "/api/ask": api_ask},
        site_dir=Path(__file__).parent / "site",
        port=8003,
    )
