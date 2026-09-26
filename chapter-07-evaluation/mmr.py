"""Maximal Marginal Relevance (MMR): pick results that are relevant AND not redundant.

Greedy: repeatedly choose the candidate d that maximises

    λ · relevance(d)  −  (1 − λ) · max_{s already selected} similarity(d, s)

λ = 1   → pure relevance (plain top-k)
λ = 0.5 → strong push for variety
Why care? If the top 5 chunks all say the same thing (e.g. three near-duplicate warranty paragraphs),
the LLM gets one fact five times and misses the other half of the question.
"""

from __future__ import annotations

import numpy as np


def mmr(relevance: np.ndarray, vectors: np.ndarray, k: int, lambda_: float = 0.7) -> list[int]:
    """relevance: (n,) scores, higher = better (min-max normalised inside).
    vectors: (n, dim) L2-normalised embeddings of the candidates. Returns chosen indices in order."""
    n = len(relevance)
    if n == 0:
        return []
    rel = (relevance - relevance.min()) / (np.ptp(relevance) or 1.0)
    sim = vectors @ vectors.T
    selected: list[int] = []
    remaining = list(range(n))
    while remaining and len(selected) < k:
        if selected:
            redundancy = sim[np.ix_(remaining, selected)].max(axis=1)
        else:
            redundancy = np.zeros(len(remaining))
        mmr_scores = lambda_ * rel[remaining] - (1 - lambda_) * redundancy
        best = remaining[int(np.argmax(mmr_scores))]
        selected.append(best)
        remaining.remove(best)
    return selected
