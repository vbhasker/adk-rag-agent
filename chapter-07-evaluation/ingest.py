"""Ingestion (load → chunk → [contextualise] → embed → store) and index management.

    python ingest.py                                   # build the default index
    python ingest.py --context none --query "E-12"     # index bare chunks, no heading context
    python ingest.py --context llm                     # needs contexts from `python contextualize.py`

Settings from args or env: CHUNK_STRATEGY, CHUNK_SIZE, CHUNK_OVERLAP, CONTEXT_MODE (none|header|llm).
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
CONTEXTS_FILE = Path(__file__).parent / ".cache" / "contexts.json"
DEFAULTS = {
    "strategy": os.getenv("CHUNK_STRATEGY", "markdown"),
    "size": int(os.getenv("CHUNK_SIZE", "120")),
    "overlap": int(os.getenv("CHUNK_OVERLAP", "20")),
}


def default_context_mode() -> str:
    mode = os.getenv("CONTEXT_MODE", "auto")
    if mode == "auto":
        return "llm" if CONTEXTS_FILE.exists() else "header"
    return mode


def load_contexts() -> dict[str, str]:
    """chunk-text hash → context sentence(s), written by contextualize.py."""
    return json.loads(CONTEXTS_FILE.read_text()) if CONTEXTS_FILE.exists() else {}


def chunk_key(chunk: Chunk) -> str:
    return hashlib.sha1(f"{chunk.doc_id}\n{chunk.text}".encode()).hexdigest()


def build_store(chunks: list[Chunk], name: str, mode: str) -> VectorStore:
    """Embed `chunks` (as text_for_embedding(mode)) into a persisted store; reuse it if nothing changed."""
    embedder = get_embedder()
    texts = [c.text_for_embedding(mode) for c in chunks]
    fingerprint = hashlib.sha1("\n".join([embedder.name, mode, *texts]).encode()).hexdigest()
    directory = INDEX_DIR / name
    store = VectorStore.load(directory, fingerprint)
    if store is None:
        print(f"[ingest] embedding {len(chunks)} chunks for index '{name}' with {embedder.name}…")
        store = VectorStore(embedder.name, embedder.dim)
        store.add(chunks, embedder.embed_documents(texts))
        store.save(directory, fingerprint)
    return store


def build_index(
    strategy: str = DEFAULTS["strategy"],
    size: int = DEFAULTS["size"],
    overlap: int = DEFAULTS["overlap"],
    mode: str | None = None,
) -> VectorStore:
    mode = mode or default_context_mode()
    chunks = chunk_corpus(load_documents(), strategy, size, overlap)
    if mode == "llm":
        contexts = load_contexts()
        if not contexts:
            raise RuntimeError("CONTEXT_MODE=llm but no contexts found. Run `python contextualize.py` first.")
        for chunk in chunks:
            chunk.context = contexts.get(chunk_key(chunk), "")
    return build_store(chunks, f"{strategy}-{size}-{overlap}-{mode}", mode)


_STORES: dict[str, VectorStore] = {}


def get_store(mode: str | None = None) -> VectorStore:
    mode = mode or default_context_mode()
    if mode not in _STORES:
        _STORES[mode] = build_index(mode=mode)
    return _STORES[mode]


def chunk_to_dict(chunk: Chunk, score: float) -> dict[str, Any]:
    return {
        "source_id": chunk.id,
        "title": chunk.title,
        "section": chunk.section,
        "category": chunk.category,
        "score": round(score, 4),
        "text": chunk.text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", choices=["fixed", "sentence", "markdown"], default=DEFAULTS["strategy"])
    parser.add_argument("--size", type=int, default=DEFAULTS["size"])
    parser.add_argument("--overlap", type=int, default=DEFAULTS["overlap"])
    parser.add_argument("--context", choices=["none", "header", "llm"], default=None)
    parser.add_argument("--query", help="run a vector query against the index")
    args = parser.parse_args()

    store = build_index(args.strategy, args.size, args.overlap, args.context)
    print(f"[ingest] {len(store.chunks)} chunks · {store.embedder_name} · context mode {args.context or default_context_mode()}")
    if args.query:
        for chunk, score in store.search(get_embedder().embed_query(args.query), top_k=4):
            print(f"  {score:.4f}  [{chunk.id}] {chunk.section}")


if __name__ == "__main__":
    main()
