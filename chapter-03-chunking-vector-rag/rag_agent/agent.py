"""Chapter 3: the first complete RAG loop as a Google ADK agent.

Run with the ADK dev UI from the chapter folder:   adk web      (then pick "rag_agent")
or from the terminal:                              python ask.py "Is a speed chip covered by warranty?"

The pattern: retrieval is a *tool*. The LLM decides when to search, reads the returned chunks, and
answers with citations. (Chapter 8 makes this loop much smarter.)
"""

from __future__ import annotations

import sys
from pathlib import Path

CHAPTER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CHAPTER_DIR))  # so the agent can import ingest.py when started via `adk web`

from dotenv import load_dotenv  # noqa: E402

load_dotenv(CHAPTER_DIR / ".env")

from llm_config import agent_model  # noqa: E402
from google.adk.agents import Agent  # noqa: E402

from ingest import retrieve  # noqa: E402

MODEL = agent_model()  # Gemini by default; LLM_PROVIDER=openai switches to OpenAI via LiteLlm


def search_knowledge_base(query: str, category: str = "") -> dict:
    """Search the Nimbus Bikes knowledge base (product specs, battery care, troubleshooting,
    error codes, warranty, returns, shipping, safety rules).

    Args:
        query: A focused search query, e.g. "battery storage temperature winter".
        category: Optional filter: "product", "support", "troubleshooting" or "policy". Leave empty to search everything.

    Returns:
        A dict with "results": a list of chunks, each with a source_id to cite, title, section and text.
    """
    results = retrieve(query, top_k=4, category=category or None)
    return {"query": query, "results": results}


INSTRUCTION = """You are the friendly support assistant for Nimbus Bikes, an e-bike company.

How to answer:
1. ALWAYS call `search_knowledge_base` before answering a question about Nimbus products, policies or problems.
   Search again with different words if the first results don't contain the answer.
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
