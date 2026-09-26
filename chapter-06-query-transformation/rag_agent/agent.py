"""Chapter 6: the RAG agent does its own query transformation.

Key insight: in a tool-calling agent, the LLM *already writes the search query*, so query rewriting
(resolving "the cargo one" from chat history) comes for free. We go one step further: the tool accepts
a LIST of queries, so Gemini does multi-query + decomposition in the same call, with no extra LLM
round-trip. pipeline.retrieve() fuses them with RRF, optionally expands to parent sections
(PARENT_DOCS=true) and adds HyDE (HYDE_IN_TOOL=true), then reranks.

Run:  adk web   (pick rag_agent)   or   python ask.py "..."
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

CHAPTER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CHAPTER_DIR))  # so the agent can import ingest.py when started via `adk web`

from dotenv import load_dotenv  # noqa: E402

load_dotenv(CHAPTER_DIR / ".env")

from google.adk.agents import Agent  # noqa: E402

from pipeline import retrieve  # noqa: E402

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def search_knowledge_base(queries: list[str], category: str = "") -> dict:
    """Search the Nimbus Bikes knowledge base (product specs, battery care, troubleshooting,
    error codes, warranty, returns, shipping, safety rules). Hybrid search: exact terms AND meaning.

    Args:
        queries: 1 to 4 standalone search queries. Resolve references from the conversation
            ("the cargo one" -> "Nimbus Cumulus cargo bike"). If the question has several parts,
            write one query per part. Otherwise add 1-2 differently-worded variants
            (e.g. casual + technical wording). Keep error codes and product names exact.
        category: Optional filter: "product" (specs, prices, sizes), "support" (battery care, maintenance,
            app), "troubleshooting" (error codes, won't turn on) or "policy" (warranty, returns, shipping,
            laws). Only set it when ALL queries belong to that category; otherwise leave empty.

    Returns:
        A dict with "results": chunks with a source_id to cite, title, section and text.
    """
    queries = [q for q in queries if q.strip()][:4] or ["Nimbus Bikes"]
    return {"queries": queries, "results": retrieve(queries, top_k=6, category=category or None)}


INSTRUCTION = """You are the friendly support assistant for Nimbus Bikes, an e-bike company.

How to answer:
1. ALWAYS call `search_knowledge_base` before answering a question about Nimbus products, policies or problems.
   Pass several queries: one per part of a multi-part question, rewritten to be standalone (resolve
   "it", "that one", "the other model" from the conversation). Search again if something is still missing.
2. Answer ONLY from the retrieved text. Never invent specs, prices, time limits or policies.
3. Cite every fact with its source id in square brackets, e.g. "The battery is covered for 2 years [warranty#1]."
4. If the knowledge base doesn't contain the answer, say so plainly and suggest contacting help@nimbusbikes.example.
5. Be concise: a short direct answer first, then details or steps if useful.
"""

root_agent = Agent(
    name="nimbus_support_agent",
    model=MODEL,
    description="Answers Nimbus Bikes customer questions using retrieval-augmented generation.",
    instruction=INSTRUCTION,
    tools=[search_knowledge_base],
)
