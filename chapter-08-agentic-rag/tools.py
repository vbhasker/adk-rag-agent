"""The retrieval toolbox shared by both agents. Plain Python functions: ADK turns the signature and
docstring into the tool schema Gemini sees, so the docstrings are written FOR the model.

    list_documents        a catalog to plan with ("which docs could answer this?")
    search_knowledge_base the Chapter 1-6 retrieval stack: multi-query → hybrid → RRF → rerank
    get_document          read a whole document when chunks aren't enough (small-to-big, agent-driven)
"""

from __future__ import annotations

from corpus import load_documents
from pipeline import retrieve

DOCS = {d.id: d for d in load_documents()}


def list_documents() -> dict:
    """List every document in the Nimbus Bikes knowledge base with its id, title and category.
    Use it to plan which documents might answer a question, or to confirm a topic is NOT covered.
    """
    return {"documents": [{"doc_id": d.id, "title": d.title, "category": d.category} for d in DOCS.values()]}


def search_knowledge_base(queries: list[str], category: str = "") -> dict:
    """Hybrid search (exact terms + meaning, reranked) over the Nimbus Bikes knowledge base.

    Args:
        queries: 1 to 4 standalone search queries. One per sub-question for multi-part questions,
            otherwise the question plus 1-2 differently-worded variants. Keep error codes and
            product names exact.
        category: Optional filter: "product", "support", "troubleshooting" or "policy".
            Only set it when every query belongs to that category.

    Returns:
        "results": chunks with a source_id to cite (e.g. "warranty#2"), title, section and text.
    """
    queries = [q for q in queries if q.strip()][:4] or ["Nimbus Bikes"]
    return {"queries": queries, "results": retrieve(queries, top_k=6, category=category or None)}


def get_document(doc_id: str) -> dict:
    """Read an entire document by its doc_id (e.g. "warranty"). Use it when a search result looks
    cut off, or when you need the surrounding sections. Cite facts from it as [doc_id].
    """
    doc = DOCS.get(doc_id)
    if doc is None:
        return {"error": f"Unknown doc_id {doc_id!r}", "available": sorted(DOCS)}
    return {"doc_id": doc.id, "title": doc.title, "category": doc.category, "text": doc.text}
