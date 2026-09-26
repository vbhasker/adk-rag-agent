"""Hybrid search: run BM25 (Chapter 1) and vector search (Chapters 2-3) over the SAME chunks,
then fuse the two ranked lists into one.

Two fusion methods:

  Reciprocal Rank Fusion (RRF), the 2026 default:
        score(chunk) = Σ over lists  1 / (k + rank_in_list)          k = 60 by convention
    Uses only RANKS, so it doesn't care that BM25 scores look like 7.3 and cosine like 0.64.
    No tuning, no normalisation, hard to break.

  Weighted (convex) combination:
        score(chunk) = α · norm(vector_score) + (1 − α) · norm(bm25_score)
    Needs score normalisation (min-max here) and a tuned α, but can beat RRF once you have an
    evaluation set to tune on (Chapter 7).

    python hybrid.py "E-12"
    python hybrid.py "my bike won't turn on" --fusion weighted --alpha 0.7
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any

from bm25 import BM25, tokenize
from chunking import Chunk
from embeddings import get_embedder
from ingest import default_context_mode, get_store
from vector_store import VectorStore


@dataclass
class Candidate:
    chunk: Chunk
    score: float = 0.0
    bm25_rank: int | None = None      # 1-based rank in the BM25 list (None = not retrieved)
    vector_rank: int | None = None
    bm25_score: float = 0.0
    vector_score: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.chunk.id,
            "title": self.chunk.title,
            "section": self.chunk.section,
            "category": self.chunk.category,
            "score": round(self.score, 4),
            "bm25_rank": self.bm25_rank,
            "vector_rank": self.vector_rank,
            "text": self.chunk.text,
            **self.extra,
        }


def rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion over any number of ranked id lists."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return fused


def min_max(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    return {key: (v - lo) / (hi - lo) if hi > lo else 1.0 for key, v in scores.items()}


class HybridRetriever:
    def __init__(self, store: VectorStore, mode: str = "header"):
        self.store, self.mode = store, mode
        self.by_id = {c.id: c for c in store.chunks}
        # BM25 indexes the same text we embed (per context mode), so both retrievers see the same content.
        self.bm25 = BM25([tokenize(c.text_for_embedding(mode)) for c in store.chunks])

    def bm25_search(self, query: str, n: int, category: str | None = None) -> list[tuple[Chunk, float]]:
        hits = self.bm25.search(tokenize(query), top_k=len(self.store.chunks))
        chunks = [(self.store.chunks[h.index], h.score) for h in hits]
        return [(c, s) for c, s in chunks if not category or c.category == category][:n]

    def vector_search(self, query: str, n: int, category: str | None = None, as_document: bool = False) -> list[tuple[Chunk, float]]:
        where = {"category": category} if category else None
        embedder = get_embedder()
        # HyDE embeds a hypothetical *answer*, so it's embedded as a document, not as a query.
        vector = embedder.embed_documents([query])[0] if as_document else embedder.embed_query(query)
        return self.store.search(vector, n, where)

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",          # hybrid | bm25 | vector
        fusion: str = "rrf",           # rrf | weighted
        alpha: float = 0.5,            # weighted only: 1.0 = all vector, 0.0 = all BM25
        rrf_k: int = 60,
        candidates: int = 20,          # how deep to look in each list before fusing
        category: str | None = None,
        vector_text: str | None = None,  # Chapter 6 HyDE: embed this hypothetical answer instead of the query
    ) -> list[Candidate]:
        bm25_hits = self.bm25_search(query, candidates, category) if mode in {"hybrid", "bm25"} else []
        vector_hits = []
        if mode in {"hybrid", "vector"}:
            vector_hits = self.vector_search(vector_text or query, candidates, category, as_document=bool(vector_text))

        pool: dict[str, Candidate] = {}
        for rank, (chunk, score) in enumerate(bm25_hits, 1):
            cand = pool.setdefault(chunk.id, Candidate(chunk))
            cand.bm25_rank, cand.bm25_score = rank, score
        for rank, (chunk, score) in enumerate(vector_hits, 1):
            cand = pool.setdefault(chunk.id, Candidate(chunk))
            cand.vector_rank, cand.vector_score = rank, score

        if mode == "bm25":
            fused = {cid: c.bm25_score for cid, c in pool.items()}
        elif mode == "vector":
            fused = {cid: c.vector_score for cid, c in pool.items()}
        elif fusion == "rrf":
            fused = rrf([[c.id for c, _ in bm25_hits], [c.id for c, _ in vector_hits]], k=rrf_k)
        else:
            v = min_max({c.id: s for c, s in vector_hits})
            b = min_max({c.id: s for c, s in bm25_hits})
            fused = {cid: alpha * v.get(cid, 0.0) + (1 - alpha) * b.get(cid, 0.0) for cid in pool}

        for cid, score in fused.items():
            pool[cid].score = score
        return sorted(pool.values(), key=lambda c: c.score, reverse=True)[:top_k]


_RETRIEVERS: dict[str, HybridRetriever] = {}


def get_retriever(mode: str | None = None) -> HybridRetriever:
    """One retriever per context mode (none | header | llm); default from CONTEXT_MODE."""
    mode = mode or default_context_mode()
    if mode not in _RETRIEVERS:
        _RETRIEVERS[mode] = HybridRetriever(get_store(mode), mode)
    return _RETRIEVERS[mode]


def retrieve(query: str, top_k: int = 4, category: str | None = None) -> list[dict[str, Any]]:
    """The retrieval function the agent uses from this chapter on: hybrid + RRF."""
    return [c.to_dict() for c in get_retriever().search(query, top_k, category=category)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query")
    parser.add_argument("--fusion", choices=["rrf", "weighted"], default="rrf")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    retriever = get_retriever()
    for mode in ("bm25", "vector", "hybrid"):
        print(f"\n  {mode.upper()}{f' ({args.fusion})' if mode == 'hybrid' else ''}")
        for rank, c in enumerate(retriever.search(args.query, args.top_k, mode, args.fusion, args.alpha), 1):
            ranks = f"bm25 #{c.bm25_rank or '-'} · vec #{c.vector_rank or '-'}"
            print(f"   #{rank} {c.score:8.4f}  [{c.chunk.id}]  {c.chunk.section or c.chunk.title}   ({ranks})")


if __name__ == "__main__":
    main()
