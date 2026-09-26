"""Contextual retrieval: ask Gemini to write 1-2 sentences that situate each chunk in its document,
and prepend them to the chunk before indexing (both BM25 and embeddings).

Why: a chunk like "Hold the power button and the minus button together for 5 seconds" doesn't say
it's about *clearing error codes on the Nimbus display*. The generated context adds exactly that.
Anthropic popularised this in 2024 ("contextual retrieval") and reported large drops in retrieval
failures, especially combined with BM25 and reranking. That's the stack we've built.

Cost control: one LLM call per chunk, done ONCE at ingestion, cached in .cache/contexts.json
(keyed by chunk text, so only new or changed chunks are re-processed). With long documents, use
Gemini context caching so the full document isn't re-billed for every chunk.

    python contextualize.py            # needs GOOGLE_API_KEY or Vertex AI
    CONTEXT_MODE=llm python server.py  # (auto mode picks it up automatically once the file exists)
"""

from __future__ import annotations

import json
import sys

import llm
from chunking import chunk_corpus
from corpus import load_documents
from ingest import CONTEXTS_FILE, DEFAULTS, chunk_key, load_contexts

PROMPT = """<document>
{document}
</document>

Here is a chunk from the document above:
<chunk>
{chunk}
</chunk>

Write one or two short sentences that situate this chunk within the overall document, to improve
search retrieval of the chunk. Mention the product, policy or topic it belongs to and what question it
answers. Answer ONLY with the context sentences, nothing else."""


def main() -> None:
    if not llm.has_llm_credentials():
        sys.exit("Contextual retrieval needs Gemini. Set GOOGLE_API_KEY (or Vertex AI) in .env first.")
    docs = {d.id: d for d in load_documents()}
    chunks = chunk_corpus(list(docs.values()), DEFAULTS["strategy"], DEFAULTS["size"], DEFAULTS["overlap"])
    contexts = load_contexts()
    todo = [c for c in chunks if chunk_key(c) not in contexts]
    print(f"{len(chunks)} chunks, {len(todo)} need context (model {llm.model_name()})")
    for i, chunk in enumerate(todo, 1):
        doc = docs[chunk.doc_id]
        contexts[chunk_key(chunk)] = llm.generate_text(PROMPT.format(document=doc.text, chunk=chunk.text), temperature=0)
        print(f"  [{i}/{len(todo)}] {chunk.id}: {contexts[chunk_key(chunk)][:90]}…")
        CONTEXTS_FILE.parent.mkdir(exist_ok=True)
        CONTEXTS_FILE.write_text(json.dumps(contexts, indent=1))  # save as we go
    print(f"Saved {len(contexts)} contexts to {CONTEXTS_FILE}")


if __name__ == "__main__":
    main()
