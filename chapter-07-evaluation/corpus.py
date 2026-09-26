"""Load the Nimbus Bikes knowledge base from data/docs/*.md.

Each file has a tiny front-matter header (id, title, category) followed by Markdown.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DOCS_DIR = Path(__file__).parent / "data" / "docs"


@dataclass
class Document:
    id: str
    title: str
    category: str
    text: str  # Markdown body without the front matter


def _parse(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    meta: dict[str, str] = {}
    body = raw
    if raw.startswith("---"):
        _, header, body = raw.split("---", 2)
        for line in header.strip().splitlines():
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    return Document(
        id=meta.get("id", path.stem),
        title=meta.get("title", path.stem),
        category=meta.get("category", "general"),
        text=body.strip(),
    )


def load_documents(docs_dir: Path = DOCS_DIR) -> list[Document]:
    return [_parse(p) for p in sorted(docs_dir.glob("*.md"))]
