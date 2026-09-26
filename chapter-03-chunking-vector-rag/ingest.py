"""Ingestion (load → chunk → embed → store) and retrieval, the "R" in RAG.

    python ingest.py                          # build (or reuse) the index and print stats
    python ingest.py --strategy fixed --size 60 --overlap 10
    python ingest.py --query "is a speed chip covered by warranty?"

Settings come from args or env vars: CHUNK_STRATEGY, CHUNK_SIZE, CHUNK_OVERLAP.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from chunking import Chunk, chunk_corpus  # noqa: E402
from corpus import load_documents  # noqa: E402
from embeddings import get_embedder  # noqa: E402
from vector_store import VectorStore  # noqa: E402

INDEX_DIR = Path(__file__).parent / ".index"
DEFAULTS = {
    "strategy": os.getenv("CHUNK_STRATEGY", "markdown"),
    "size": int(os.getenv("CHUNK_SIZE", "120")),
    "overlap": int(os.getenv("CHUNK_OVERLAP", "20")),
}


def build_index(strategy: str = DEFAULTS["strategy"], size: int = DEFAULTS["size"], overlap: int = DEFAULTS["overlap"]) -> VectorStore:
    embedder = get_embedder()
    chunks = chunk_corpus(load_documents(), strategy, size, overlap)
    texts = [c.text_for_embedding() for c in chunks]
    fingerprint = hashlib.sha1("\n".join([embedder.name, *texts]).encode()).hexdigest()
    directory = INDEX_DIR / f"{strategy}-{size}-{overlap}"

    store = VectorStore.load(directory, fingerprint)
    if store is None:
        print(f"[ingest] embedding {len(chunks)} chunks ({strategy}, size={size}, overlap={overlap}) with {embedder.name}…")
        store = VectorStore(embedder.name, embedder.dim)
        store.add(chunks, embedder.embed_documents(texts))
        store.save(directory, fingerprint)
    return store


_STORE: VectorStore | None = None


def get_store() -> VectorStore:
    global _STORE
    if _STORE is None:
        _STORE = build_index()
    return _STORE


def chunk_to_dict(chunk: Chunk, score: float) -> dict[str, Any]:
    return {
        "source_id": chunk.id,
        "title": chunk.title,
        "section": chunk.section,
        "category": chunk.category,
        "score": round(score, 4),
        "text": chunk.text,
    }


def retrieve(query: str, top_k: int = 4, category: str | None = None) -> list[dict[str, Any]]:
    store = get_store()
    query_vector = get_embedder().embed_query(query)
    hits = store.search(query_vector, top_k, where={"category": category} if category else None)
    return [chunk_to_dict(c, s) for c, s in hits]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", choices=["fixed", "sentence", "markdown"], default=DEFAULTS["strategy"])
    parser.add_argument("--size", type=int, default=DEFAULTS["size"])
    parser.add_argument("--overlap", type=int, default=DEFAULTS["overlap"])
    parser.add_argument("--query", help="run a retrieval query against the index")
    args = parser.parse_args()

    store = build_index(args.strategy, args.size, args.overlap)
    lengths = [len(c.text.split()) for c in store.chunks]
    print(f"[ingest] {len(store.chunks)} chunks · avg {sum(lengths) / len(lengths):.0f} words · min {min(lengths)} · max {max(lengths)} · {store.embedder_name}")
    if args.query:
        embedder = get_embedder()
        for chunk, score in store.search(embedder.embed_query(args.query), top_k=4):
            print(f"\n  {score:.4f}  [{chunk.id}] {chunk.title} > {chunk.section}\n    {chunk.text[:200]}…")


if __name__ == "__main__":
    main()
