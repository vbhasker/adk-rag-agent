"""Second-stage reranking: re-score the top ~20 hybrid candidates with a model that reads the
query and each passage TOGETHER.

    Bi-encoder (Chapters 2-4): embed query and passage separately, compare vectors.
        Fast (passage vectors are precomputed) but the two texts never "see" each other.
    Cross-encoder (this chapter): feed "[query] [SEP] [passage]" through one transformer → one score.
        Much more accurate (it can notice negation, which entity is meant, whether the passage
        actually ANSWERS the question) but must run once per candidate at query time. Too slow for
        a million docs, perfect for re-sorting 20.

Rerankers (RERANKER=auto|cross-encoder|llm|none):
    cross-encoder  Xenova/ms-marco-MiniLM-L-6-v2 via fastembed, local CPU, ~80 MB, no key
    llm            an LLM (Gemini or OpenAI, per LLM_PROVIDER) grades every candidate 0-10 in one
                   structured-output call (needs a key). "gemini" is accepted as an alias.
    none           keep the fused order (baseline)
"""

from __future__ import annotations

import math
import os

from pydantic import BaseModel

import llm


class Reranker:
    name = "none"

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Return one relevance score per passage (higher = more relevant)."""
        raise NotImplementedError


class NoReranker(Reranker):
    name = "none"

    def score(self, query: str, passages: list[str]) -> list[float]:
        return [1.0 - i / max(1, len(passages)) for i in range(len(passages))]  # keep incoming order


class CrossEncoderReranker(Reranker):
    def __init__(self, model: str | None = None):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model_name = model or os.getenv("RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
        self.model = TextCrossEncoder(model_name=self.model_name)
        self.name = f"cross-encoder:{self.model_name}"

    def score(self, query: str, passages: list[str]) -> list[float]:
        logits = list(self.model.rerank(query, passages))
        return [1 / (1 + math.exp(-x)) for x in logits]  # sigmoid → 0..1, easier to read and threshold


class _Grade(BaseModel):
    id: int
    score: float  # 0 (irrelevant) .. 10 (directly answers the question)


class _Grades(BaseModel):
    grades: list[_Grade]


class LlmReranker(Reranker):
    """LLM-as-reranker: one call grades all candidates. Slower and pricier than a cross-encoder,
    but understands nuance and works in any language."""

    def __init__(self):
        self.name = f"llm:{llm.provider()}:{llm.model_name()}"

    def score(self, query: str, passages: list[str]) -> list[float]:
        numbered = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages))
        prompt = (
            "You are a search relevance grader. For each passage, rate from 0 to 10 how well it helps "
            "answer the question (10 = directly answers it, 5 = related but incomplete, 0 = irrelevant).\n"
            f"Return a grade for every id from 0 to {len(passages) - 1}.\n\n"
            f"Question: {query}\n\nPassages:\n{numbered}"
        )
        grades = {g.id: g.score / 10 for g in llm.generate_json(prompt, _Grades).grades}
        return [grades.get(i, 0.0) for i in range(len(passages))]


_CACHE: dict[str, Reranker] = {}


def load_reranker(kind: str) -> Reranker:
    """Build (once) a reranker by kind: "cross-encoder", "llm" or "none". May raise if unavailable."""
    kind = "llm" if kind == "gemini" else kind  # older name for the LLM reranker
    if kind not in _CACHE:
        _CACHE[kind] = {"cross-encoder": CrossEncoderReranker, "llm": LlmReranker, "none": NoReranker}[kind]()
    return _CACHE[kind]


_AVAILABLE: list[str] | None = None


def available_rerankers() -> list[str]:
    """Which rerankers can run here (checked once; model downloads are slow)."""
    global _AVAILABLE
    if _AVAILABLE is None:
        _AVAILABLE = ["none"]
        try:
            load_reranker("cross-encoder")
            _AVAILABLE.append("cross-encoder")
        except Exception as exc:  # fastembed missing or model download blocked
            print(f"[rerank] cross-encoder unavailable ({type(exc).__name__}: {exc})")
        if llm.has_llm_credentials():
            _AVAILABLE.append("llm")
    return _AVAILABLE


def get_reranker() -> Reranker:
    """The default reranker, chosen once by RERANKER=auto|cross-encoder|llm|none."""
    if "default" not in _CACHE:
        choice = os.getenv("RERANKER", "auto").lower()
        if choice == "auto":
            available = available_rerankers()
            choice = "cross-encoder" if "cross-encoder" in available else "llm" if "llm" in available else "none"
        _CACHE["default"] = load_reranker(choice)
        print(f"[rerank] using {_CACHE['default'].name}")
    return _CACHE["default"]
