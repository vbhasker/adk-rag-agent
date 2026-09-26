"""Chapter 1 CLI: keyword search with BM25 (and TF-IDF for comparison).

Usage:
    python search.py "error E-07"
    python search.py "how do I charge the battery" --algo tfidf
    python search.py                      # interactive mode
"""

from __future__ import annotations

import argparse

from bm25 import BM25, TfIdf, tokenize
from corpus import load_documents


def build(algo: str = "bm25", k1: float = 1.5, b: float = 0.75):
    docs = load_documents()
    corpus_tokens = [tokenize(f"{d.title}\n{d.text}") for d in docs]
    index = TfIdf(corpus_tokens) if algo == "tfidf" else BM25(corpus_tokens, k1=k1, b=b)
    return docs, index


def run_query(query: str, algo: str, k1: float, b: float, top_k: int) -> None:
    docs, index = build(algo, k1, b)
    terms = tokenize(query)
    print(f"\nQuery: {query!r}  ->  tokens {terms}")
    for term in dict.fromkeys(terms):
        print(f"   idf({term}) = {index.idf(term):.3f}   (in {index.df.get(term, 0)}/{index.n_docs} docs)")
    results = index.search(terms, top_k)
    if not results:
        print("\n  No document shares a single token with the query. Keyword search found nothing!")
    for rank, r in enumerate(results, 1):
        d = docs[r.index]
        parts = ", ".join(f"{t}={v:.2f}" for t, v in sorted(r.breakdown.items(), key=lambda kv: -kv[1]))
        print(f"\n  #{rank}  {r.score:6.3f}  {d.title}  [{d.category}]\n        because: {parts}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query", nargs="?", help="search query (omit for interactive mode)")
    parser.add_argument("--algo", choices=["bm25", "tfidf"], default="bm25")
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if args.query:
        run_query(args.query, args.algo, args.k1, args.b, args.top_k)
        return
    print("Interactive BM25 search. Try: 'error E-07', 'bike will not turn on', 'refund'. Empty line quits.")
    while query := input("\nsearch> ").strip():
        run_query(query, args.algo, args.k1, args.b, args.top_k)


if __name__ == "__main__":
    main()
