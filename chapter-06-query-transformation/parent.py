"""Parent-document retrieval ("small-to-big"): search SMALL chunks, return their BIG parent.

Small chunks (one or two sentences) embed precisely: one idea per vector, so they match sharply.
But they're too small to answer from ("It takes about 10 minutes."). So we index small children,
and when a child matches we hand the LLM its whole parent section instead.

    parent  = a Chapter 3 markdown chunk (a section, ~45 words avg here, up to 120)
    child   = ~30-word sentence packs inside that parent, id "warranty#2.1"
"""

from __future__ import annotations

from chunking import Chunk, sentence_pack
from hybrid import HybridRetriever
from ingest import build_store, default_context_mode, get_store

_CHILD_RETRIEVERS: dict[str, HybridRetriever] = {}


def child_chunks(parents: list[Chunk], size: int = 30) -> list[Chunk]:
    children = []
    for parent in parents:
        for k, text in enumerate(sentence_pack(parent.text, size, overlap_sentences=0)):
            children.append(Chunk(id=f"{parent.id}.{k}", doc_id=parent.doc_id, title=parent.title, section=parent.section,
                                  category=parent.category, text=text, position=k, context=parent.context))
    return children


def get_child_retriever(mode: str | None = None) -> HybridRetriever:
    mode = mode or default_context_mode()
    if mode not in _CHILD_RETRIEVERS:
        children = child_chunks(get_store(mode).chunks)
        _CHILD_RETRIEVERS[mode] = HybridRetriever(build_store(children, f"children-{mode}", mode), mode)
    return _CHILD_RETRIEVERS[mode]


def parent_id(child_id: str) -> str:
    return child_id.rsplit(".", 1)[0]
