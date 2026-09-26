"""Chapter 1 lesson site + playground API.  Run:  python server.py  ->  http://localhost:8001"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bm25 import BM25, TfIdf, tokenize
from corpus import load_documents
from webserver import serve

DOCS = load_documents()
CORPUS_TOKENS = [tokenize(f"{d.title}\n{d.text}") for d in DOCS]


def api_search(p: dict[str, Any]) -> dict[str, Any]:
    query = str(p.get("query", ""))
    algo = p.get("algo", "bm25")
    k1, b = float(p.get("k1", 1.5)), float(p.get("b", 0.75))
    stem, stop = bool(p.get("stem", True)), bool(p.get("stopWords", True))

    corpus_tokens = [tokenize(f"{d.title}\n{d.text}", stop, stem) for d in DOCS]
    index = TfIdf(corpus_tokens) if algo == "tfidf" else BM25(corpus_tokens, k1=k1, b=b)
    terms = tokenize(query, stop, stem)
    results = index.search(terms, top_k=int(p.get("topK", 5)))
    return {
        "tokens": terms,
        "idf": {t: {"idf": round(index.idf(t), 3), "df": index.df.get(t, 0)} for t in dict.fromkeys(terms)},
        "avgdl": round(index.avgdl, 1),
        "results": [
            {
                "id": DOCS[r.index].id,
                "title": DOCS[r.index].title,
                "category": DOCS[r.index].category,
                "length": index.doc_len[r.index],
                "score": round(r.score, 4),
                "breakdown": {t: round(v, 4) for t, v in r.breakdown.items()},
                "snippet": DOCS[r.index].text[:220],
            }
            for r in results
        ],
    }


def api_tokenize(p: dict[str, Any]) -> dict[str, Any]:
    text = str(p.get("text", ""))
    return {
        "raw": tokenize(text, remove_stop_words=False, stem=False),
        "noStopWords": tokenize(text, remove_stop_words=True, stem=False),
        "stemmed": tokenize(text, remove_stop_words=True, stem=True),
    }


def api_stats(_: dict[str, Any]) -> dict[str, Any]:
    index = BM25(CORPUS_TOKENS)
    by_idf = sorted(index.df, key=index.idf)
    return {
        "documents": len(DOCS),
        "vocabulary": len(index.df),
        "avgdl": round(index.avgdl, 1),
        "commonTerms": [{"term": t, "df": index.df[t], "idf": round(index.idf(t), 3)} for t in by_idf[:8]],
        "rareTerms": [{"term": t, "df": index.df[t], "idf": round(index.idf(t), 3)} for t in by_idf[-8:]],
    }


if __name__ == "__main__":
    serve(
        {"/api/search": api_search, "/api/tokenize": api_tokenize, "/api/stats": api_stats},
        site_dir=Path(__file__).parent / "site",
        port=8001,
    )
