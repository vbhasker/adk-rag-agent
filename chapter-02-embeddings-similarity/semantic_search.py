"""Chapter 2: semantic search over whole documents, side by side with BM25.

Usage:
    python semantic_search.py "how long does the battery last"
    python semantic_search.py "E07" --metric euclidean
    python semantic_search.py --compare "my bike won't turn on" "the display stays dark"
"""

from __future__ import annotations

import argparse

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from bm25 import BM25, tokenize  # noqa: E402
from corpus import Document, load_documents  # noqa: E402
from embeddings import Embedder, get_embedder  # noqa: E402
from similarity import all_measures, scores  # noqa: E402


class SemanticIndex:
    """One vector per document. (Chapter 3 shows why one vector per *chunk* works better.)"""

    def __init__(self, docs: list[Document], embedder: Embedder):
        self.docs, self.embedder = docs, embedder
        self.matrix = embedder.embed_documents([f"{d.title}\n{d.text}" for d in docs])

    def search(self, query: str, metric: str = "cosine", top_k: int = 5) -> list[tuple[Document, float]]:
        q = self.embedder.embed_query(query)
        s = scores(q, self.matrix, metric)
        order = np.argsort(-s)[:top_k]
        return [(self.docs[i], float(s[i])) for i in order]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query", nargs="?", default="how long does the battery last")
    parser.add_argument("--metric", choices=["cosine", "dot", "euclidean", "manhattan"], default="cosine")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--compare", nargs=2, metavar=("TEXT_A", "TEXT_B"), help="compare two sentences")
    args = parser.parse_args()

    embedder = get_embedder()
    if args.compare:
        a, b = (embedder.embed_query(t) for t in args.compare)
        for name, value in all_measures(a, b).items():
            print(f"  {name:10s} {value: .4f}")
        return

    docs = load_documents()
    index = SemanticIndex(docs, embedder)
    bm25 = BM25([tokenize(f"{d.title}\n{d.text}") for d in docs])

    print(f"\nQuery: {args.query!r}\n\n  SEMANTIC ({embedder.name}, {args.metric})")
    for rank, (doc, score) in enumerate(index.search(args.query, args.metric, args.top_k), 1):
        print(f"   #{rank} {score: .4f}  {doc.title}")
    print("\n  BM25 (Chapter 1)")
    for rank, r in enumerate(bm25.search(tokenize(args.query), args.top_k), 1):
        print(f"   #{rank} {r.score: .4f}  {docs[r.index].title}")


if __name__ == "__main__":
    main()
