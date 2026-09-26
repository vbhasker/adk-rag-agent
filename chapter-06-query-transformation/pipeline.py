"""The Chapter 6 retrieval pipeline. Everything from Chapters 3-5, plus query- and chunk-side upgrades:

    [query transforms]  one or more queries (multi-query / decomposition) + optional HyDE passage
        → hybrid search per query (over chunks indexed with context mode none | header | llm)
        → RRF across all the lists
        → [parent-document] swap matched small children for their parent sections
        → rerank against the original question → threshold → MMR → top k

    python pipeline.py "can my kid ride on the back and is it legal in the EU?" --multi
    python pipeline.py "how do I clear stored codes" --parent --context none
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any

import numpy as np

from hybrid import Candidate, get_retriever, rrf
from ingest import default_context_mode, get_store
from mmr import mmr
from parent import get_child_retriever, parent_id
from query_transform import hyde, multi_query
from rerank import get_reranker, load_reranker


def run_pipeline(
    query: str,
    top_k: int = 5,
    candidates: int = 20,
    use_rerank: bool = True,
    mmr_lambda: float | None = None,
    min_score: float | None = None,
    category: str | None = None,
    reranker_kind: str | None = None,
    queries: list[str] | None = None,     # multi-query: search each, fuse with RRF (default: [query])
    hyde_passage: str | None = None,      # HyDE: add a vector search using this hypothetical answer
    context_mode: str | None = None,      # none | header | llm
    parent_docs: bool = False,            # small-to-big retrieval
) -> dict[str, Any]:
    mode = context_mode or default_context_mode()
    retriever = get_child_retriever(mode) if parent_docs else get_retriever(mode)
    reranker = load_reranker(reranker_kind) if reranker_kind else get_reranker()
    timings: dict[str, float] = {}

    # 1. One hybrid search per query (+ one HyDE vector search), then RRF across all lists.
    t0 = time.perf_counter()
    lists = [retriever.search(q, top_k=candidates, category=category) for q in (queries or [query])]
    if hyde_passage:
        lists.append(retriever.search(query, candidates, mode="vector", category=category, vector_text=hyde_passage))
    pool = {c.chunk.id: c for results in lists for c in reversed(results)}
    fused_scores = rrf([[c.chunk.id for c in results] for results in lists])
    fused = sorted(pool.values(), key=lambda c: fused_scores[c.chunk.id], reverse=True)
    for c in fused:
        c.score = fused_scores[c.chunk.id]

    # 2. Small-to-big: replace child chunks by their parent section (keep the best-ranked child per parent).
    if parent_docs:
        parents = {c.id: c for c in get_store(mode).chunks}
        seen: dict[str, Candidate] = {}
        for c in fused:
            pid = parent_id(c.chunk.id)
            if pid not in seen:
                seen[pid] = Candidate(parents[pid], score=c.score, extra={"matched_child": c.chunk.text})
        fused = list(seen.values())
    fused = fused[:candidates]
    timings["retrieve_ms"] = (time.perf_counter() - t0) * 1000
    for rank, cand in enumerate(fused, 1):
        cand.extra["fused_rank"] = rank
    fused_view = [c.to_dict() for c in fused[:top_k]]

    # 3. Rerank against the ORIGINAL question (not the rewrites).
    ranked = fused
    if use_rerank and fused:
        t0 = time.perf_counter()
        scores = reranker.score(query, [c.chunk.text_for_embedding(mode) for c in fused])
        timings["rerank_ms"] = (time.perf_counter() - t0) * 1000
        for cand, score in zip(fused, scores):
            cand.extra["rerank_score"] = round(float(score), 4)
            cand.score = float(score)
        ranked = sorted(fused, key=lambda c: c.score, reverse=True)
        if min_score is not None:
            ranked = [c for c in ranked if c.score >= min_score]

    # 4. Optional diversity.
    if mmr_lambda is not None and ranked:
        store = get_store(mode)
        rows = {c.id: i for i, c in enumerate(store.chunks)}
        vectors = store.matrix[[rows[c.chunk.id] for c in ranked]]
        final = [ranked[i] for i in mmr(np.array([c.score for c in ranked]), vectors, top_k, mmr_lambda)]
    else:
        final = ranked[:top_k]

    return {
        "reranker": reranker.name if use_rerank else "off",
        "contextMode": mode,
        "queries": queries or [query],
        "timings": {k: round(v, 1) for k, v in timings.items()},
        "fused": fused_view,
        "final": [c.to_dict() for c in final],
    }


def retrieve(queries: list[str], top_k: int = 5, category: str | None = None) -> list[dict[str, Any]]:
    """What the agent calls. The agent writes the query variants itself (free multi-query!);
    we fuse them, expand to parent sections if PARENT_DOCS=true, and rerank against the first query."""
    min_score = os.getenv("RERANK_MIN_SCORE")
    hyde_passage = hyde(queries[0])["passage"] if os.getenv("HYDE_IN_TOOL", "false").lower() == "true" else None
    return run_pipeline(
        queries[0], top_k=top_k, category=category, queries=queries, hyde_passage=hyde_passage,
        min_score=float(min_score) if min_score else None,
        parent_docs=os.getenv("PARENT_DOCS", "false").lower() == "true",
    )["final"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query")
    parser.add_argument("--multi", action="store_true", help="multi-query / decomposition")
    parser.add_argument("--hyde", action="store_true", help="add a HyDE search (needs an LLM key)")
    parser.add_argument("--parent", action="store_true", help="small-to-big parent-document retrieval")
    parser.add_argument("--context", choices=["none", "header", "llm"], default=None)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    queries = multi_query(args.query) if args.multi else {"queries": [args.query], "source": "off"}
    passage = hyde(args.query) if args.hyde else {"passage": None, "source": "off"}
    print(f"queries ({queries['source']}): {queries['queries']}\nhyde ({passage['source']}): {passage['passage']}")
    out = run_pipeline(args.query, args.top_k, queries=queries["queries"], hyde_passage=passage["passage"],
                       context_mode=args.context, parent_docs=args.parent)
    print(f"\nreranker: {out['reranker']} · context: {out['contextMode']} · {out['timings']}")
    for i, c in enumerate(out["final"], 1):
        child = f"\n        matched child: {c['matched_child'][:90]}…" if "matched_child" in c else ""
        print(f"   #{i} {c['score']:.4f} [{c['source_id']}] {c['section']}{child}")


if __name__ == "__main__":
    main()
