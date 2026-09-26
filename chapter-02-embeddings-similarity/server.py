"""Chapter 2 lesson site + playground API.  Run:  python server.py  ->  http://localhost:8002"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from bm25 import BM25, tokenize  # noqa: E402
from corpus import load_documents  # noqa: E402
from embeddings import get_embedder  # noqa: E402
from semantic_search import SemanticIndex  # noqa: E402
from similarity import all_measures  # noqa: E402
from webserver import serve  # noqa: E402

DOCS = load_documents()
EMBEDDER = get_embedder()
INDEX = SemanticIndex(DOCS, EMBEDDER)
BM25_INDEX = BM25([tokenize(f"{d.title}\n{d.text}") for d in DOCS])


def api_info(_: dict[str, Any]) -> dict[str, Any]:
    return {"embedder": EMBEDDER.name, "dim": EMBEDDER.dim, "isToy": EMBEDDER.name.startswith("toy")}


def api_compare(p: dict[str, Any]) -> dict[str, Any]:
    a, b = EMBEDDER.embed_query(str(p["a"])), EMBEDDER.embed_query(str(p["b"]))
    return {
        "measures": {k: round(v, 4) for k, v in all_measures(a, b).items()},
        "previewA": [round(float(x), 3) for x in a[:12]],
        "previewB": [round(float(x), 3) for x in b[:12]],
        "dim": int(a.shape[0]),
    }


def api_search(p: dict[str, Any]) -> dict[str, Any]:
    query, metric, top_k = str(p.get("query", "")), p.get("metric", "cosine"), int(p.get("topK", 5))
    semantic = [
        {"id": d.id, "title": d.title, "category": d.category, "score": round(s, 4)}
        for d, s in INDEX.search(query, metric, top_k)
    ]
    keyword = [
        {"id": DOCS[r.index].id, "title": DOCS[r.index].title, "category": DOCS[r.index].category, "score": round(r.score, 4)}
        for r in BM25_INDEX.search(tokenize(query), top_k)
    ]
    return {"semantic": semantic, "bm25": keyword}


def api_map(p: dict[str, Any]) -> dict[str, Any]:
    """Project the 768/384-dim vectors down to 2D with PCA so we can *see* the space."""
    query = str(p.get("query", ""))
    q = EMBEDDER.embed_query(query)
    points = np.vstack([INDEX.matrix, q])
    centered = points - points[:-1].mean(axis=0)  # PCA fitted on the documents only
    _, _, vt = np.linalg.svd(centered[:-1], full_matrices=False)
    xy = centered @ vt[:2].T
    sims = INDEX.matrix @ q
    return {
        "docs": [
            {"id": d.id, "title": d.title, "category": d.category, "x": float(x), "y": float(y), "sim": round(float(s), 4)}
            for d, (x, y), s in zip(DOCS, xy[:-1], sims)
        ],
        "query": {"x": float(xy[-1][0]), "y": float(xy[-1][1]), "text": query},
    }


if __name__ == "__main__":
    serve(
        {"/api/info": api_info, "/api/compare": api_compare, "/api/search": api_search, "/api/map": api_map},
        site_dir=Path(__file__).parent / "site",
        port=8002,
    )
