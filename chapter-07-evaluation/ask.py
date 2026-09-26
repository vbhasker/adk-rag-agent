"""Ask the Chapter 3 RAG agent a question from the terminal (needs a Gemini key or Vertex AI).

    python ask.py "Is a speed chip covered by the warranty?"
"""

from __future__ import annotations

import json
import sys

from agent_runner import ask, has_llm_credentials
from rag_agent.agent import root_agent


def main() -> None:
    question = " ".join(sys.argv[1:]) or "How should I store my battery over the winter?"
    if not has_llm_credentials():
        sys.exit("Set GOOGLE_API_KEY (or Vertex AI settings) in .env first. See .env.example.")
    result = ask(root_agent, question)
    for step in result["trace"]:
        if step["type"] == "tool_call":
            print(f"🔎 {step['name']}({json.dumps(step['args'])})")
        elif step["type"] == "tool_result":
            ids = [r["source_id"] for r in step["response"].get("results", [])]
            print(f"   ↳ retrieved {ids}")
    print(f"\n{result['answer']}")


if __name__ == "__main__":
    main()
