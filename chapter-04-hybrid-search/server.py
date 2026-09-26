"""Chapter 4 lesson site + playground API.  Run:  python server.py  ->  http://localhost:8004"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_runner import ask, has_llm_credentials
from hybrid import get_retriever
from webserver import serve

# A tiny labelled set: (query, doc that answers it). Chapter 7 turns this idea into a proper evaluation.
SCOREBOARD = [
    ("E-12", "error-codes"),
    ("E07", "error-codes"),
    ("UN3480", "returns-shipping"),
    ("firmware 4.2.1", "app-connectivity"),
    ("my bike won't turn on", "troubleshooting-power"),
    ("how long does the battery last", "battery-care"),
    ("can I send it back if I don't like it", "returns-shipping"),
    ("is a speed chip covered", "warranty"),
    ("what happens if the motor gets too hot", "error-codes"),
    ("which bike can carry two kids", "model-cumulus-cargo"),
    ("I'm 170 cm tall, which frame size?", "sizing-fit"),
    ("how fast can an e-bike go in Europe", "safety-regulations"),
]


def api_info(_: dict[str, Any]) -> dict[str, Any]:
    r = get_retriever()
    return {"embedder": r.store.embedder_name, "chunks": len(r.store.chunks), "llm": has_llm_credentials()}


def _settings(p: dict[str, Any]) -> dict[str, Any]:
    return {"fusion": p.get("fusion", "rrf"), "alpha": float(p.get("alpha", 0.5)), "rrf_k": int(p.get("rrfK", 60))}


def api_search(p: dict[str, Any]) -> dict[str, Any]:
    r, query, top_k = get_retriever(), str(p.get("query", "")), int(p.get("topK", 5))
    return {mode: [c.to_dict() for c in r.search(query, top_k, mode, **_settings(p))] for mode in ("bm25", "vector", "hybrid")}


def api_scoreboard(p: dict[str, Any]) -> dict[str, Any]:
    """Hit@3 and MRR per mode: did the right document show up in the top 3 chunks, and how high?"""
    r, rows = get_retriever(), []
    totals = {m: {"hits": 0, "mrr": 0.0} for m in ("bm25", "vector", "hybrid")}
    for query, expected in SCOREBOARD:
        row: dict[str, Any] = {"query": query, "expected": expected}
        for mode in totals:
            docs = [c.chunk.doc_id for c in r.search(query, 10, mode, **_settings(p))]
            rank = docs.index(expected) + 1 if expected in docs else None
            row[mode] = rank
            totals[mode]["hits"] += int(rank is not None and rank <= 3)
            totals[mode]["mrr"] += 1 / rank if rank else 0
        rows.append(row)
    n = len(SCOREBOARD)
    return {"rows": rows, "summary": {m: {"hitAt3": t["hits"] / n, "mrr": round(t["mrr"] / n, 3)} for m, t in totals.items()}}


def api_ask(p: dict[str, Any]) -> dict[str, Any]:
    if not has_llm_credentials():
        return {"error": "No LLM credentials. Add GOOGLE_API_KEY (or Vertex AI settings) to .env, or set LLM_PROVIDER=openai with OPENAI_API_KEY, then restart."}
    from rag_agent.agent import root_agent

    return ask(root_agent, str(p.get("question", "")))


if __name__ == "__main__":
    get_retriever()
    serve(
        {"/api/info": api_info, "/api/search": api_search, "/api/scoreboard": api_scoreboard, "/api/ask": api_ask},
        site_dir=Path(__file__).parent / "site",
        port=8004,
    )
