"""The two-stage retrieval pipeline used from Chapter 5 on:

    hybrid search (recall: top 20 candidates, cheap)
        → metadata filter (optional, applied inside both retrievers)
        → rerank (precision: cross-encoder / Gemini re-sorts the 20)
        → relevance threshold (optional: drop weak candidates so the agent can say "I don't know")
        → MMR (optional: diversity)
        → top k to the LLM

    python pipeline.py "what happens if the motor gets too hot"
    python pipeline.py "warranty" --mmr 0.5
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any

import numpy as np

from hybrid import Candidate, get_retriever
from mmr import mmr
from rerank import get_reranker, load_reranker


def run_pipeline(
    query: str,
    top_k: int = 5,
    candidates: int = 20,
    use_rerank: bool = True,
    mmr_lambda: float | None = None,      # None = no MMR
    min_score: float | None = None,       # drop reranked candidates scoring below this (0..1)
    category: str | None = None,
    reranker_kind: str | None = None,     # override the default: "cross-encoder" | "gemini" | "none"
) -> dict[str, Any]:
    retriever = get_retriever()
    reranker = load_reranker(reranker_kind) if reranker_kind else get_reranker()
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    fused = retriever.search(query, top_k=candidates, category=category)
    timings["hybrid_ms"] = (time.perf_counter() - t0) * 1000
    for rank, cand in enumerate(fused, 1):
        cand.extra["fused_rank"] = rank
    fused_view = [c.to_dict() for c in fused[:top_k]]  # snapshot before reranking changes the scores

    ranked: list[Candidate] = fused
    if use_rerank and fused:
        t0 = time.perf_counter()
        scores = reranker.score(query, [c.chunk.text_for_embedding() for c in fused])
        timings["rerank_ms"] = (time.perf_counter() - t0) * 1000
        for cand, score in zip(fused, scores):
            cand.extra["rerank_score"] = round(float(score), 4)
            cand.score = float(score)
        ranked = sorted(fused, key=lambda c: c.score, reverse=True)
        if min_score is not None:
            ranked = [c for c in ranked if c.score >= min_score]

    if mmr_lambda is not None and ranked:
        rows = {c.id: i for i, c in enumerate(retriever.store.chunks)}
        vectors = retriever.store.matrix[[rows[c.chunk.id] for c in ranked]]
        order = mmr(np.array([c.score for c in ranked]), vectors, top_k, mmr_lambda)
        final = [ranked[i] for i in order]
    else:
        final = ranked[:top_k]

    return {
        "reranker": reranker.name if use_rerank else "off",
        "timings": {k: round(v, 1) for k, v in timings.items()},
        "fused": fused_view,
        "final": [c.to_dict() for c in final],
    }


def retrieve(query: str, top_k: int = 5, category: str | None = None) -> list[dict[str, Any]]:
    """What the agent calls: hybrid → rerank → (optional threshold) → top k.

    Set RERANK_MIN_SCORE (e.g. 0.2 for the cross-encoder) to drop weak chunks, so that for
    off-topic questions the agent receives nothing and says "I don't know" instead of improvising.
    """
    min_score = os.getenv("RERANK_MIN_SCORE")
    return run_pipeline(query, top_k=top_k, category=category, min_score=float(min_score) if min_score else None)["final"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mmr", type=float, default=None, help="MMR lambda, e.g. 0.5")
    parser.add_argument("--min-score", type=float, default=None)
    args = parser.parse_args()

    out = run_pipeline(args.query, args.top_k, mmr_lambda=args.mmr, min_score=args.min_score)
    print(f"\nreranker: {out['reranker']}   timings: {out['timings']}")
    print("\n  BEFORE (hybrid order)")
    for i, c in enumerate(out["fused"], 1):
        print(f"   #{i} [{c['source_id']}] {c['section']}")
    print(f"\n  AFTER (reranked{' + MMR' if args.mmr is not None else ''})")
    for i, c in enumerate(out["final"], 1):
        print(f"   #{i} {c.get('rerank_score', 0):.3f} [{c['source_id']}] {c['section']}   (was #{c['fused_rank']})")


if __name__ == "__main__":
    main()
