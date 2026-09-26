"""Chunking: split documents into retrieval-sized pieces.

Three strategies, from naive to what most 2026 production systems start with:

    fixed     every N words, with overlap. Simple; happily cuts sentences and sections in half.
    sentence  pack whole sentences up to N words, overlapping by a sentence. Never cuts mid-sentence.
    markdown  structure-aware + recursive: split on headings first, keep the heading path
              ("Warranty Policy > What is not covered") as metadata, and only split an oversized
              section further by paragraphs, then sentences. This is the default.

Sizes are in *words* to keep things dependency-free; 1 word ≈ 1.3 tokens for English.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from corpus import Document

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9*\"'(])|\n(?=\s*[-*\d])")


@dataclass
class Chunk:
    id: str           # "warranty#2": doc id + position, used as a citation handle
    doc_id: str
    title: str
    section: str      # heading path inside the doc ("" for fixed/sentence strategies)
    category: str
    text: str
    position: int

    def text_for_embedding(self) -> str:
        """What we actually embed: the chunk plus light context about where it came from."""
        header = f"{self.title} > {self.section}" if self.section else self.title
        return f"{header}\n{self.text}"


def _words(text: str) -> list[str]:
    return text.split()


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]


def fixed_size(text: str, size: int = 120, overlap: int = 20) -> list[str]:
    words = _words(text)
    step = max(1, size - overlap)
    return [" ".join(words[i : i + size]) for i in range(0, max(1, len(words) - overlap), step)]


def sentence_pack(text: str, size: int = 120, overlap_sentences: int = 1) -> list[str]:
    sentences, chunks, current = _sentences(text), [], []
    for sentence in sentences:
        if current and len(_words(" ".join(current + [sentence]))) > size:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
        current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks


def _split_sections(markdown: str) -> list[tuple[str, str]]:
    """Return (heading, body) pairs for each '## ' section; text before the first one is the intro."""
    sections: list[tuple[str, str]] = []
    heading, lines = "Introduction", []
    for line in markdown.splitlines():
        if line.startswith("# "):  # document title, already stored as metadata
            continue
        if line.startswith("## "):
            if "".join(lines).strip():
                sections.append((heading, "\n".join(lines).strip()))
            heading, lines = line[3:].strip(), []
        else:
            lines.append(line)
    if "".join(lines).strip():
        sections.append((heading, "\n".join(lines).strip()))
    return sections


def _recursive_split(text: str, size: int, overlap: int) -> list[str]:
    """Paragraphs first; only fall back to sentence packing for paragraphs that are still too big."""
    if len(_words(text)) <= size:
        return [text]
    pieces, current = [], ""
    for para in [p.strip() for p in text.split("\n\n") if p.strip()]:
        if len(_words(para)) > size:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(sentence_pack(para, size, overlap_sentences=1 if overlap else 0))
        elif len(_words(f"{current}\n\n{para}")) > size:
            pieces.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}".strip()
    if current:
        pieces.append(current)
    return pieces


def chunk_document(doc: Document, strategy: str = "markdown", size: int = 120, overlap: int = 20) -> list[Chunk]:
    if strategy == "fixed":
        parts = [("", t) for t in fixed_size(doc.text, size, overlap)]
    elif strategy == "sentence":
        parts = [("", t) for t in sentence_pack(doc.text, size, overlap_sentences=1 if overlap else 0)]
    elif strategy == "markdown":
        parts = [(heading, piece) for heading, body in _split_sections(doc.text) for piece in _recursive_split(body, size, overlap)]
    else:
        raise ValueError(f"unknown chunking strategy {strategy!r}")
    return [
        Chunk(id=f"{doc.id}#{i}", doc_id=doc.id, title=doc.title, section=section, category=doc.category, text=text, position=i)
        for i, (section, text) in enumerate(parts)
    ]


def chunk_corpus(docs: list[Document], strategy: str = "markdown", size: int = 120, overlap: int = 20) -> list[Chunk]:
    return [chunk for doc in docs for chunk in chunk_document(doc, strategy, size, overlap)]
