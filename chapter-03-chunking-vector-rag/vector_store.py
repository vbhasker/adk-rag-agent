"""A minimal vector store: a NumPy matrix + metadata + exact (brute-force) search, saved to disk.

What real vector databases add on top of this:
  * ANN indexes (HNSW graphs, IVF clusters, Google's ScaNN) → sub-linear search over millions of vectors,
    trading a little recall for a lot of speed. Below ~100k vectors, exact search like this is fine.
  * Metadata filtering at scale, updates/deletes, replication, access control.
Production options: pgvector / AlloyDB, Vertex AI Vector Search, Qdrant, Weaviate, Milvus, LanceDB, Chroma…
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from chunking import Chunk


class VectorStore:
    def __init__(self, embedder_name: str, dim: int):
        self.embedder_name, self.dim = embedder_name, dim
        self.chunks: list[Chunk] = []
        self.matrix = np.zeros((0, dim), dtype=np.float32)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        assert vectors.shape == (len(chunks), self.dim), "one normalised vector per chunk"
        self.chunks.extend(chunks)
        self.matrix = np.vstack([self.matrix, vectors.astype(np.float32)])

    def search(self, query_vector: np.ndarray, top_k: int = 5, where: dict[str, str] | None = None) -> list[tuple[Chunk, float]]:
        """Exact nearest neighbours by dot product (= cosine, vectors are normalised).

        `where` is a metadata filter, e.g. {"category": "policy"}: applied *before* ranking
        (pre-filtering), so you always get top_k results from the allowed subset.
        """
        if not self.chunks:
            return []
        mask = np.ones(len(self.chunks), dtype=bool)
        for key, value in (where or {}).items():
            mask &= np.array([getattr(c, key) == value for c in self.chunks])
        scores = np.where(mask, self.matrix @ query_vector, -np.inf)
        k = min(top_k, int(mask.sum()))
        top = np.argpartition(-scores, k - 1)[:k] if k else []  # O(n) partial sort, then sort only k
        top = sorted(top, key=lambda i: -scores[i])
        return [(self.chunks[i], float(scores[i])) for i in top]

    # ---- persistence -------------------------------------------------------------------------
    def save(self, directory: Path, fingerprint: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.matrix)
        meta = {"embedder": self.embedder_name, "dim": self.dim, "fingerprint": fingerprint, "chunks": [asdict(c) for c in self.chunks]}
        (directory / "chunks.json").write_text(json.dumps(meta, indent=1))

    @classmethod
    def load(cls, directory: Path, fingerprint: str) -> "VectorStore | None":
        meta_path = directory / "chunks.json"
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text())
        if meta.get("fingerprint") != fingerprint:  # model or chunking changed → rebuild
            return None
        store = cls(meta["embedder"], meta["dim"])
        store.chunks = [Chunk(**c) for c in meta["chunks"]]
        store.matrix = np.load(directory / "vectors.npy")
        return store
