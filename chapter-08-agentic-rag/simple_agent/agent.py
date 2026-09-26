"""The Chapter 6 single-agent RAG, kept here as the baseline to compare against rag_agent/.

One LlmAgent, one search tool, one pass: search (maybe twice) → answer. Fast and cheap.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

CHAPTER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CHAPTER_DIR))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(CHAPTER_DIR / ".env")

from google.adk.agents import Agent  # noqa: E402

from tools import search_knowledge_base  # noqa: E402

root_agent = Agent(
    name="simple_rag_agent",
    model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    description="Single-agent RAG baseline for Nimbus Bikes.",
    instruction="""You are the friendly support assistant for Nimbus Bikes, an e-bike company.

1. ALWAYS call `search_knowledge_base` before answering. Pass one query per part of the question,
   rewritten to be standalone. Search again if something is still missing.
2. Answer ONLY from the retrieved text. Cite every fact with its source id, e.g. [warranty#1].
3. If the knowledge base doesn't contain the answer, say so and suggest help@nimbusbikes.example.
4. Be concise: direct answer first, then details.""",
    tools=[search_knowledge_base],
)
