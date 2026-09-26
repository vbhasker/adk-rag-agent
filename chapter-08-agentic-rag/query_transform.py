"""Query transformations: fix the question before searching.

    rewrite      Turn a follow-up ("what about the cargo one?") + chat history into a standalone query.
    multi_query  Generate several differently-worded queries, and split multi-part questions into
                 one query per part. Each is searched separately; results are fused with RRF.
    hyde         Hypothetical Document Embeddings: let the LLM write a fake answer passage, then search
                 with THAT. Answers look like answers, so they land closer to real answer chunks than
                 a short question does. (The fake answer's facts don't matter. Only its "shape" does.)

All three use the configured LLM (Gemini by default, OpenAI with LLM_PROVIDER=openai). Without
credentials we fall back to simple heuristics (clearly labelled), so the pipeline still runs offline.

    python query_transform.py "can my kid ride on the back and is it legal in the EU?"
"""

from __future__ import annotations

import re
import sys

from pydantic import BaseModel

import llm


class _Query(BaseModel):
    query: str


class _Queries(BaseModel):
    queries: list[str]


def rewrite(query: str, history: list[str] | None = None) -> dict:
    if not history:
        return {"query": query, "source": "unchanged (no history)"}
    if not llm.has_llm_credentials():
        return {"query": f"{history[-1]} {query}", "source": "heuristic: prepended last turn"}
    prompt = (
        "Rewrite the user's latest message as a standalone search query for an e-bike company's help "
        "center, resolving pronouns and references using the conversation. Keep product names, error "
        "codes and numbers exactly as written.\n\n"
        + "\n".join(f"Earlier user message: {h}" for h in history)
        + f"\nLatest message: {query}"
    )
    return {"query": llm.generate_json(prompt, _Query).query, "source": f"llm:{llm.model_name()}"}


def multi_query(query: str, n: int = 3) -> dict:
    if not llm.has_llm_credentials():
        # Heuristic decomposition: split on question marks / " and " between clauses.
        parts = [p.strip(" ?.,") for p in re.split(r"\?\s*|\s+and\s+(?=(?:is|are|can|do|does|how|what|which|when|where|why)\b)", query) if p.strip(" ?.,")]
        return {"queries": list(dict.fromkeys([query, *parts])), "source": "heuristic split (no LLM)"}
    prompt = (
        f"Generate up to {n} search queries for an e-bike company's knowledge base that together retrieve "
        "everything needed to answer the question. If it has several parts, write one query per part. "
        "Otherwise write differently-worded variants (synonyms, technical vs. casual wording). Keep error "
        f"codes and product names exact.\n\nQuestion: {query}"
    )
    queries = llm.generate_json(prompt, _Queries).queries[:n]
    return {"queries": list(dict.fromkeys([query, *queries])), "source": f"llm:{llm.model_name()}"}


def hyde(query: str) -> dict:
    if not llm.has_llm_credentials():
        return {"passage": None, "source": "unavailable (needs an LLM key)"}
    prompt = (
        "Write a short passage (60-100 words) in the style of an e-bike company's help-center article "
        f"that answers this question. Invent plausible details if needed.\n\nQuestion: {query}"
    )
    return {"passage": llm.generate_text(prompt, temperature=0.3), "source": f"llm:{llm.model_name()}"}


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "can my kid ride on the back and is it legal in the EU?"
    print("multi_query:", multi_query(q))
    print("hyde:", hyde(q))
    print("rewrite:", rewrite("what about the cargo one?", ["How much does the Stratus weigh?"]))
