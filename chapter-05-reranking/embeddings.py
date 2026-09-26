"""Turn text into vectors (embeddings). Four interchangeable providers:

    gemini : Google's gemini-embedding-001 via the google-genai SDK (needs GOOGLE_API_KEY or Vertex AI)
    openai : OpenAI text-embedding-3-small/large via the openai SDK (needs OPENAI_API_KEY)
    local  : BAAI/bge-small-en-v1.5 running on your CPU via fastembed (no key; ~70 MB download once)
    toy    : a hashing trick over words + character trigrams. NOT semantic, but runs anywhere offline.

Pick one with EMBEDDINGS_PROVIDER=gemini|openai|local|toy (default "auto": gemini if a Google key is set,
else openai if OPENAI_API_KEY is set, else local, else toy). Every vector is L2-normalised, so cosine similarity == dot product (see similarity.py).

Embeddings are cached on disk (.cache/) so you never pay twice to embed the same text.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

CACHE_DIR = Path(__file__).parent / ".cache"


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


class Embedder:
    """Interface. Documents and queries are embedded differently on purpose: many models are
    *asymmetric* (a short question should land near the long passage that answers it)."""

    name = "base"
    dim = 0

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        raise NotImplementedError


class GeminiEmbedder(Embedder):
    def __init__(self, model: str | None = None, dim: int | None = None):
        from google import genai  # imported lazily so other providers work without it

        self.client = genai.Client()  # reads GOOGLE_API_KEY, or GOOGLE_GENAI_USE_VERTEXAI + project/location
        self.model = model or os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
        # Matryoshka embeddings: 3072 dims by default, can be truncated to 1536 / 768 with little quality loss.
        self.dim = dim or int(os.getenv("EMBEDDING_DIM", "768"))
        self.name = f"gemini:{self.model}@{self.dim}"

    def _embed(self, texts: list[str], task_type: str) -> np.ndarray:
        from google.genai import types

        config = types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dim)
        out: list[list[float]] = []
        for start in range(0, len(texts), 50):  # batch to stay under request limits
            batch = texts[start : start + 50]
            result = self.client.models.embed_content(model=self.model, contents=batch, config=config)
            out.extend(e.values for e in result.embeddings)
        return normalize(np.array(out, dtype=np.float32))  # truncated dims must be re-normalised

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


class OpenAIEmbedder(Embedder):
    """OpenAI embeddings. Symmetric: OpenAI has no query/document task types, so both use the same call.
    text-embedding-3-* models are also Matryoshka-style: OPENAI_EMBEDDING_DIM can shorten them."""

    KNOWN_DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072, "text-embedding-ada-002": 1536}

    def __init__(self, model: str | None = None):
        from openai import OpenAI  # imported lazily so other providers work without it

        self.client = OpenAI()  # reads OPENAI_API_KEY (and OPENAI_BASE_URL for compatible endpoints)
        self.model = model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        requested = os.getenv("OPENAI_EMBEDDING_DIM")
        self.requested_dim = int(requested) if requested else None
        self.dim = self.requested_dim or self.KNOWN_DIMS.get(self.model) or len(self._embed(["probe"])[0])
        self.name = f"openai:{self.model}@{self.dim}"

    def _embed(self, texts: list[str]) -> np.ndarray:
        extra = {"dimensions": self.requested_dim} if self.requested_dim else {}
        out: list[list[float]] = []
        for start in range(0, len(texts), 100):
            response = self.client.embeddings.create(model=self.model, input=texts[start : start + 100], **extra)
            out.extend(item.embedding for item in response.data)
        return normalize(np.array(out, dtype=np.float32))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text])[0]


class LocalEmbedder(Embedder):
    def __init__(self, model: str | None = None):
        from fastembed import TextEmbedding

        self.model_name = model or os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        # LOCAL_EMBEDDING_PATH: optional folder with a pre-downloaded ONNX model (offline machines).
        self.model = TextEmbedding(model_name=self.model_name, specific_model_path=os.getenv("LOCAL_EMBEDDING_PATH"))
        self.dim = len(next(iter(self.model.query_embed("probe"))))
        self.name = f"local:{self.model_name}"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return normalize(np.array(list(self.model.passage_embed(texts)), dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return normalize(np.array(next(iter(self.model.query_embed(text))), dtype=np.float32))


class ToyEmbedder(Embedder):
    """Feature hashing: each word and character trigram is hashed into one of `dim` buckets.

    It captures spelling overlap ('charging' ~ 'charger') but has zero understanding of meaning
    ('bike won't start' is NOT close to 'display stays dark'). Handy for offline smoke tests and
    as a baseline that shows what learned embeddings add.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"toy:hashing@{dim}"

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            padded = f"#{word}#"
            for feature in [word, *(padded[i : i + 3] for i in range(len(padded) - 2))]:
                h = int(hashlib.md5(feature.encode()).hexdigest(), 16)
                vec[h % self.dim] += 1.0 if (h >> 64) % 2 else -1.0
        return vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return normalize(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return normalize(self._vector(text))


class CachedEmbedder(Embedder):
    """Wraps any embedder with a JSON disk cache keyed by (provider, kind, text)."""

    def __init__(self, inner: Embedder):
        self.inner, self.name, self.dim = inner, inner.name, inner.dim
        CACHE_DIR.mkdir(exist_ok=True)
        self.path = CACHE_DIR / f"embeddings-{re.sub(r'[^a-zA-Z0-9]+', '_', self.name)}.json"
        self.cache: dict[str, list[float]] = json.loads(self.path.read_text()) if self.path.exists() else {}

    def _key(self, kind: str, text: str) -> str:
        return hashlib.sha1(f"{kind}\n{text}".encode()).hexdigest()

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if self._key("doc", t) not in self.cache]
        if missing:
            for text, vec in zip(missing, self.inner.embed_documents(missing)):
                self.cache[self._key("doc", text)] = vec.tolist()
            self.path.write_text(json.dumps(self.cache))
        return np.array([self.cache[self._key("doc", t)] for t in texts], dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        key = self._key("query", text)
        if key not in self.cache:
            self.cache[key] = self.inner.embed_query(text).tolist()
            self.path.write_text(json.dumps(self.cache))
        return np.array(self.cache[key], dtype=np.float32)


def _has_google_credentials() -> bool:
    return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")) or os.getenv(
        "GOOGLE_GENAI_USE_VERTEXAI", ""
    ).lower() in {"1", "true"}


_EMBEDDER: Embedder | None = None


def get_embedder() -> Embedder:
    """Build (once) the embedder selected by EMBEDDINGS_PROVIDER, with graceful fallbacks."""
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    provider = os.getenv("EMBEDDINGS_PROVIDER", "auto").lower()
    if provider == "auto":
        provider = "gemini" if _has_google_credentials() else "openai" if os.getenv("OPENAI_API_KEY") else "local"

    inner: Embedder
    if provider == "gemini":
        inner = GeminiEmbedder()
    elif provider == "openai":
        inner = OpenAIEmbedder()
    elif provider == "toy":
        inner = ToyEmbedder()
    else:
        try:
            inner = LocalEmbedder()
        except Exception as exc:  # no fastembed, or model download blocked
            print(f"[embeddings] local model unavailable ({type(exc).__name__}: {exc}).")
            print("[embeddings] Falling back to the TOY hashing embedder. Results will NOT be semantic!")
            inner = ToyEmbedder()
    print(f"[embeddings] using {inner.name} ({inner.dim} dims)")
    _EMBEDDER = CachedEmbedder(inner)
    return _EMBEDDER
