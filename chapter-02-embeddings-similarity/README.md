# Chapter 2 · Embeddings & Similarity

> Builds on Chapter 1. Fixes BM25's vocabulary-mismatch problem with dense vectors.

## What you'll learn
- What an embedding is, and how contrastive training gives vectors "meaning"
- Sparse (BM25) vs dense (embedding) vectors
- Cosine, dot product, Euclidean and Manhattan, and why they rank identically on normalised vectors
- Asymmetric query/document embeddings, Matryoshka dimensions, uncalibrated scores
- Semantic search, plus a 2D "meaning map" via PCA
- Weak spots: negation, exact codes, long documents squeezed into one vector

## Run it
```bash
cd chapter-02-embeddings-similarity
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # optional: add GOOGLE_API_KEY to use Gemini embeddings

python server.py                                    # → http://localhost:8002
python semantic_search.py "how long does the battery last"
python semantic_search.py --compare "I love my bike" "I do not love my bike"
```
With no key, the local `BAAI/bge-small-en-v1.5` model downloads once (~70 MB) and runs on CPU.
If that isn't possible, the `toy` hashing embedder takes over so everything still runs (not semantic!).

## Files
| File | Purpose |
|---|---|
| `embeddings.py` | Gemini / local / toy embedders, normalisation, disk cache (**reused in every later chapter**) |
| `similarity.py` | Cosine, dot, Euclidean, Manhattan |
| `semantic_search.py` | Whole-document semantic index + CLI (with BM25 side by side) |
| `server.py` | Playground API: compare, search, PCA map |
| `bm25.py`, `corpus.py`, `webserver.py`, `data/` | Carried over from Chapter 1 |

## Exercises
1. Switch `EMBEDDINGS_PROVIDER` between `local` and `gemini` and compare the "Try this" numbers.
2. Set `EMBEDDING_DIM=3072` vs `768` with Gemini. Does the ranking change on our 12 docs?
3. Embed each doc's **title only**. Better or worse? Why?

**Next:** Chapter 3 chunks the documents, builds a vector store and wires it into a Google ADK agent.
