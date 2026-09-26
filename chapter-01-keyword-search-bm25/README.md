# Chapter 1 · Keyword Search & BM25

> Start here. No API key, no `pip install`, just Python 3.10+.

## What you'll learn
- What RAG is and why **retrieval** quality decides answer quality
- Tokenization: lowercasing, stop words, stemming (and why `E-07` must stay one token)
- The inverted index
- TF-IDF, then **BM25** (term saturation `k1`, length normalisation `b`)
- Where keyword search shines (exact codes, IDs) and where it fails (**vocabulary mismatch**)

## Run it
```bash
cd chapter-01-keyword-search-bm25
python server.py                         # lesson + playground → http://localhost:8001
python search.py "error E-07"            # CLI with a per-term score breakdown
python search.py "how long does the battery last"   # watch BM25 get it wrong
python search.py --algo tfidf "battery charge"
```

## Files
| File | Purpose |
|---|---|
| `data/docs/` | The Nimbus Bikes knowledge base (12 Markdown docs, used by every chapter) |
| `corpus.py` | Loads docs + front matter |
| `bm25.py` | Tokenizer, BM25 and TF-IDF from scratch (~100 lines) |
| `search.py` | CLI |
| `server.py`, `webserver.py` | Stdlib web server for the lesson site + JSON API |
| `site/` | Lesson page; playground in TypeScript (`app.ts` → `app.js`, rebuild with `npx -p typescript tsc -p site`) |

## Exercises
1. Search `E07` (no dash). Why zero results? Add a normalisation rule in `tokenize()` that fixes it.
2. Add a title boost: tokens from `doc.title` count double (a mini BM25F).
3. Replace `light_stem` with `nltk`'s Snowball stemmer and compare rankings.

**Next:** Chapter 2 fixes the vocabulary-mismatch problem with embeddings.
