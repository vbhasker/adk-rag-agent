"""Turn eval/golden.jsonl into an ADK eval set, so you can also run ADK's built-in evaluator:

    python make_adk_evalset.py
    adk eval rag_agent rag_agent/nimbus.evalset.json --config_file_path rag_agent/test_config.json

We build the file with ADK's own Pydantic models, so the format always matches your installed ADK.
The config uses two LLM-judged ADK metrics that map onto RAG quality:

    final_response_match_v2   does the answer semantically match the reference answer?  (≈ correctness)
    hallucinations_v1         is each sentence of the answer grounded in the context/tool output? (≈ faithfulness)
"""

from __future__ import annotations

import json
from pathlib import Path

from google.adk.evaluation.eval_case import EvalCase, Invocation
from google.adk.evaluation.eval_set import EvalSet
from google.genai import types

from eval_retrieval import load_golden

AGENT_DIR = Path(__file__).parent / "rag_agent"


def main() -> None:
    cases = [
        EvalCase(
            eval_id=g["id"],
            conversation=[
                Invocation(
                    invocation_id=f"{g['id']}-turn1",
                    user_content=types.Content(role="user", parts=[types.Part(text=g["question"])]),
                    final_response=types.Content(role="model", parts=[types.Part(text=g["reference"])]),
                )
            ],
        )
        for g in load_golden()
    ]
    eval_set = EvalSet(eval_set_id="nimbus_golden", name="Nimbus Bikes golden questions", eval_cases=cases)
    (AGENT_DIR / "nimbus.evalset.json").write_text(eval_set.model_dump_json(indent=2, exclude_none=True))

    config = {"criteria": {"final_response_match_v2": {"threshold": 0.7}, "hallucinations_v1": {"threshold": 0.8}}}
    (AGENT_DIR / "test_config.json").write_text(json.dumps(config, indent=2))
    print(f"Wrote {len(cases)} eval cases to {AGENT_DIR / 'nimbus.evalset.json'} and test_config.json")


if __name__ == "__main__":
    main()
