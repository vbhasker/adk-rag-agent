"""End-to-end answer evaluation: run the ADK agent on every golden question, then grade the answers.

Needs Gemini (the agent answers, and Gemini also judges). Per question we measure:

    faithfulness   (LLM judge, 0-1)  Is every claim in the answer supported by the RETRIEVED chunks?
                                      Low = hallucination, even if the claim happens to be true.
    correctness    (LLM judge, 0-1)  Does the answer agree with the golden reference answer?
    refused        (LLM judge)       Did the agent say it doesn't know? Required for unanswerable questions.
    citations_ok   (code)            Every [source_id] cited actually appeared in the tool results.
    context_recall (code)            Did the retrieved chunks include the relevant docs?

Plus latency and number of tool calls. Results → eval/results/answers.json.

    python eval_answers.py                      # agentic agent, all 24 questions
    python eval_answers.py --agent simple       # the single-agent baseline
    python eval_answers.py --limit 5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from statistics import mean
from typing import Any

from pydantic import BaseModel

import llm
from agent_runner import ask
from eval_retrieval import RESULTS_DIR, load_golden

JUDGE_PROMPT = """You are grading a customer-support answer produced by a RAG system.

QUESTION:
{question}

RETRIEVED CONTEXT (the only information the system was allowed to use):
{context}

REFERENCE ANSWER (written by a human expert):
{reference}

SYSTEM ANSWER:
{answer}

Grade strictly:
- faithfulness: fraction (0.0-1.0) of the factual claims in SYSTEM ANSWER that are supported by the
  RETRIEVED CONTEXT. Use 1.0 if the answer makes no factual claims (e.g. a pure "I don't know").
- correctness: 0.0-1.0, how well SYSTEM ANSWER agrees with the REFERENCE ANSWER on the facts that matter.
  If the reference says the information is not available, a correct answer must say so too.
- refused: true if the SYSTEM ANSWER says it cannot find / does not have the information.
- explanation: one short sentence."""


class Judgement(BaseModel):
    faithfulness: float
    correctness: float
    refused: bool
    explanation: str


def retrieved_chunks(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Everything the agent actually read: search results plus whole documents from get_document."""
    chunks = []
    for step in trace:
        response = step.get("response") or {} if step["type"] == "tool_result" else {}
        chunks.extend(response.get("results", []))
        if "doc_id" in response and "text" in response:
            chunks.append({"source_id": response["doc_id"], "text": response["text"]})
    return chunks


def evaluate_question(agent: Any, g: dict[str, Any], answer_key: str | None = None) -> dict[str, Any]:
    t0 = time.perf_counter()
    result = ask(agent, g["question"], answer_key)
    latency = time.perf_counter() - t0
    chunks = retrieved_chunks(result["trace"])
    context = "\n\n".join(f"[{c['source_id']}] {c['text']}" for c in chunks) or "(nothing retrieved)"
    judgement = llm.generate_json(
        JUDGE_PROMPT.format(question=g["question"], context=context, reference=g["reference"], answer=result["answer"]),
        Judgement,
    )
    cited = set(re.findall(r"\[([a-z0-9-]+(?:#[\d.]+)?)\]", result["answer"]))
    retrieved_ids = {c["source_id"] for c in chunks}
    retrieved_docs = {c["source_id"].split("#")[0] for c in chunks}
    retrieved_ids |= retrieved_docs  # citing a whole document read via get_document is valid too
    relevant = set(g["relevant_docs"])
    return {
        "id": g["id"], "type": g["type"], "question": g["question"], "answer": result["answer"],
        "faithfulness": judgement.faithfulness, "correctness": judgement.correctness, "refused": judgement.refused,
        "explanation": judgement.explanation,
        "citations_ok": cited <= retrieved_ids, "cited": sorted(cited),
        "context_recall": len(relevant & retrieved_docs) / len(relevant) if relevant else None,
        "tool_calls": sum(1 for s in result["trace"] if s["type"] == "tool_call"),
        "latency_s": round(latency, 2),
    }


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [r for r in rows if r["type"] != "unanswerable"]
    unanswerable = [r for r in rows if r["type"] == "unanswerable"]
    return {
        "faithfulness": round(mean(r["faithfulness"] for r in rows), 3),
        "correctness (answerable)": round(mean(r["correctness"] for r in answerable), 3) if answerable else None,
        "false refusals (answerable)": sum(r["refused"] for r in answerable),
        "correct refusals (unanswerable)": f"{sum(r['refused'] for r in unanswerable)}/{len(unanswerable)}",
        "citations valid": f"{sum(r['citations_ok'] for r in rows)}/{len(rows)}",
        "context recall": round(mean(r["context_recall"] for r in answerable), 3) if answerable else None,
        "avg tool calls": round(mean(r["tool_calls"] for r in rows), 2),
        "avg latency (s)": round(mean(r["latency_s"] for r in rows), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--agent", choices=["agentic", "simple"], default="agentic")
    args = parser.parse_args()
    if not llm.has_llm_credentials():
        sys.exit("Answer evaluation needs an LLM. Set GOOGLE_API_KEY, or LLM_PROVIDER=openai + OPENAI_API_KEY, in .env first.")

    from ask import load_agent

    agent, answer_key = load_agent(args.agent)
    golden = load_golden()[: args.limit]
    rows = []
    for g in golden:
        row = evaluate_question(agent, g, answer_key)
        rows.append(row)
        print(f"{g['id']} [{g['type']:12s}] faith {row['faithfulness']:.2f} · correct {row['correctness']:.2f} · refused {row['refused']!s:5s} · {row['latency_s']}s")
    summary = summarise(rows)
    print("\nSUMMARY")
    for key, val in summary.items():
        print(f"  {key:34s} {val}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"answers-{args.agent}.json"
    out.write_text(json.dumps({"model": llm.model_name(), "agent": args.agent, "summary": summary, "rows": rows}, indent=1))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
