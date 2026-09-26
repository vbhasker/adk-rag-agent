"""Similarity and distance measures, written out so you can see there's no magic.

Similarity: higher = closer.  Distance: lower = closer.

    dot(a, b)       = Σ a_i b_i
    cosine(a, b)    = dot(a, b) / (|a| |b|)          angle only, ignores length. Range [-1, 1]
    euclidean(a, b) = sqrt(Σ (a_i - b_i)^2)           straight-line distance (L2)
    manhattan(a, b) = Σ |a_i - b_i|                   city-block distance (L1)

Key fact used everywhere in RAG: if vectors are normalised to length 1 then
    cosine(a, b) == dot(a, b)      and      euclidean(a, b)^2 == 2 - 2 * cosine(a, b)
so all three produce the SAME ranking. That's why vector DBs normalise once and use the cheap dot product.
"""

from __future__ import annotations

import numpy as np


def dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def manhattan(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).sum())


# Vectorised versions: one query against a whole matrix of documents at once.
def scores(query: np.ndarray, matrix: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """Return a 'higher is better' score per row, whatever the metric."""
    if metric == "dot":
        return matrix @ query
    if metric == "cosine":
        return (matrix @ query) / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(query) + 1e-12)
    if metric == "euclidean":
        return -np.linalg.norm(matrix - query, axis=1)  # negate: smaller distance = better
    if metric == "manhattan":
        return -np.abs(matrix - query).sum(axis=1)
    raise ValueError(f"unknown metric {metric!r}")


def all_measures(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    return {"cosine": cosine(a, b), "dot": dot(a, b), "euclidean": euclidean(a, b), "manhattan": manhattan(a, b)}
