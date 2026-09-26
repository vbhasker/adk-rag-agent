"""Ask either agent from the terminal (needs an LLM key: Gemini/Vertex AI, or OpenAI).

    python ask.py "Compare the battery warranty with the battery lifespan"              # agentic
    python ask.py --agent simple "Compare the battery warranty with the battery lifespan"
"""

from __future__ import annotations

import argparse
import json
import sys

from agent_runner import ask, has_llm_credentials


def load_agent(kind: str):
    if kind == "simple":
        from simple_agent.agent import root_agent
        return root_agent, None
    from rag_agent.agent import root_agent
    return root_agent, "draft_answer"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="*")
    parser.add_argument("--agent", choices=["agentic", "simple"], default="agentic")
    args = parser.parse_args()
    if not has_llm_credentials():
        sys.exit("Set GOOGLE_API_KEY (or LLM_PROVIDER=openai + OPENAI_API_KEY) in .env first. See .env.example.")

    agent, answer_key = load_agent(args.agent)
    result = ask(agent, " ".join(args.question) or "Can my kid ride on the back, and is a throttle legal in the EU?", answer_key)
    for step in result["trace"]:
        who = f"[{step['t']:5.1f}s {step['author']}]"
        if step["type"] == "tool_call":
            print(f"{who} 🔧 {step['name']}({json.dumps(step['args'])})")
        elif step["type"] == "tool_result":
            ids = [r["source_id"] for r in (step["response"] or {}).get("results", [])]
            print(f"{who}    ↳ {ids or str(step['response'])[:100]}")
        else:
            print(f"{who} 💬 {step['text'][:160]}{'…' if len(step['text']) > 160 else ''}")
    print(f"\n=== ANSWER ({result['seconds']}s) ===\n{result['answer']}")


if __name__ == "__main__":
    main()
