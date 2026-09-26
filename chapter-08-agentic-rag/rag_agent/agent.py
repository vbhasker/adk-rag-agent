"""Chapter 8: agentic RAG as a small team of ADK agents.

    agentic_rag (SequentialAgent)
    ├── researcher        LlmAgent + tools. Plans, searches (several times if needed), reads whole docs,
    │                     writes research notes with source ids and explicit GAPS.   → state["research_notes"]
    └── write_and_check (LoopAgent, max 3 rounds)
        ├── writer        drafts the answer from the notes only (+ fixes feedback)   → state["draft_answer"]
        └── fact_checker  verifies every claim against the notes; calls exit_loop     → state["review_feedback"]
                          when the draft passes, otherwise returns a list of problems.

This is the "corrective / self-reflective RAG" pattern: retrieval, generation and verification are separate
steps, and bad drafts get sent back. Compare it with the single agent in simple_agent/ (Chapter 6).

Run:  adk web   (pick rag_agent or simple_agent)   or   python ask.py --agent agentic "..."
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

CHAPTER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CHAPTER_DIR))  # so the agent can import tools.py when started via `adk web`

from dotenv import load_dotenv  # noqa: E402

load_dotenv(CHAPTER_DIR / ".env")

from google.adk.agents import Agent, LoopAgent, SequentialAgent  # noqa: E402
from google.adk.tools import exit_loop  # noqa: E402

from tools import get_document, list_documents, search_knowledge_base  # noqa: E402

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

researcher = Agent(
    name="researcher",
    model=MODEL,
    description="Finds and collects the facts needed to answer a Nimbus Bikes question.",
    instruction="""You are the research specialist for Nimbus Bikes customer support. You do NOT answer the
customer; you collect evidence for the writer.

1. Break the question into sub-questions (a simple question has one).
2. Call `search_knowledge_base` with one query per sub-question (plus a variant if wording is casual).
3. If a result looks cut off or you need the surrounding sections, call `get_document`.
4. If nothing relevant comes back, try different wording once. Use `list_documents` to check whether the
   topic exists at all.
5. Stop after at most 4 tool calls in total.

Output ONLY research notes in this format:
FACTS:
- <one fact, with exact numbers/codes as written in the source> [source_id]
- ...
GAPS: <sub-questions the knowledge base does not answer, or "none">

If the message is small talk that needs no facts, output exactly: NO_RESEARCH_NEEDED""",
    tools=[search_knowledge_base, get_document, list_documents],
    output_key="research_notes",
)

writer = Agent(
    name="writer",
    model=MODEL,
    description="Writes the customer-facing answer from the research notes.",
    instruction="""You write the final answer for a Nimbus Bikes customer.

Rules:
- Use ONLY the facts in the research notes below. Never add outside knowledge.
- Cite every fact with its source id in square brackets, exactly as in the notes, e.g. [warranty#2].
- If the notes list GAPS, say clearly which part the knowledge base doesn't cover and suggest
  help@nimbusbikes.example.
- If the notes say NO_RESEARCH_NEEDED, just reply politely and briefly.
- Short direct answer first, then steps or details if useful. No preamble.

Fact-checker feedback on your previous draft (fix every point; empty on the first round):
{review_feedback?}

Research notes:
{research_notes}""",
    output_key="draft_answer",
)

fact_checker = Agent(
    name="fact_checker",
    model=MODEL,
    description="Verifies the draft answer against the research notes.",
    instruction="""You are a strict fact checker. Compare the draft answer with the research notes.

Check that:
1. Every factual claim in the draft is supported by a note, and cites that note's source id.
2. Numbers, codes, prices and time limits match the notes exactly.
3. Every part of the customer's question is either answered or explicitly marked as not covered.

If ALL checks pass, call the `exit_loop` tool and output nothing else.
Otherwise output a short numbered list of the problems to fix (do not rewrite the answer yourself).

Research notes:
{research_notes}

Draft answer:
{draft_answer}""",
    tools=[exit_loop],
    output_key="review_feedback",
)

root_agent = SequentialAgent(
    name="agentic_rag",
    description="Research → write → fact-check loop for grounded Nimbus Bikes answers.",
    sub_agents=[
        researcher,
        LoopAgent(name="write_and_check", sub_agents=[writer, fact_checker], max_iterations=int(os.getenv("MAX_REVIEW_ROUNDS", "3"))),
    ],
)
