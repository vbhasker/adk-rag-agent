"""Retrieval metrics, computed at the DOCUMENT level.

The retriever returns chunks; we map them to their doc ids (keeping first occurrence) so metrics don't
depend on how the corpus was chunked, and different chunking strategies can be compared fairly.

    Hit@k        1 if ANY relevant doc is in the top k, else 0              "did we find something?"
    Recall@k     relevant docs found in top k / all relevant docs          "did we find everything?"
    Precision@k  relevant docs in top k / k                                "how much noise came with it?"
    MRR          1 / rank of the first relevant doc (0 if none)            "how high was the first hit?"
    nDCG@k       DCG / ideal DCG, with DCG = Σ rel_i / log2(i + 1)         "were ALL the relevant docs near the top?"
"""

from __future__ import annotations

import math


def dedupe_docs(chunk_doc_ids: list[str]) -> list[str]:
    return list(dict.fromkeys(chunk_doc_ids))


def hit_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return float(any(d in relevant for d in ranked[:k]))


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(relevant & set(ranked[:k])) / len(relevant) if relevant else 0.0


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(relevant & set(ranked[:k])) / k


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    return next((1 / i for i, d in enumerate(ranked, 1) if d in relevant), 0.0)


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1 / math.log2(i + 1) for i, d in enumerate(ranked[:k], 1) if d in relevant)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def all_metrics(ranked: list[str], relevant: set[str], k: int) -> dict[str, float]:
    return {
        f"hit@{k}": hit_at_k(ranked, relevant, k),
        f"recall@{k}": recall_at_k(ranked, relevant, k),
        f"precision@{k}": precision_at_k(ranked, relevant, k),
        "mrr": reciprocal_rank(ranked, relevant),
        f"ndcg@{k}": ndcg_at_k(ranked, relevant, k),
    }
