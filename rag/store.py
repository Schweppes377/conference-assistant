"""Persistent ChromaDB storage with deterministic, idempotent updates."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import chromadb

from rag.models import SearchResult, TextChunk


class VectorStore:
    """Small wrapper around one persistent cosine-distance collection."""

    def __init__(self, persist_dir: Path, collection_name: str, model_name: str) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "embedding_model": model_name},
        )
        stored_model = (self._collection.metadata or {}).get("embedding_model")
        if stored_model and stored_model != model_name and self._collection.count() > 0:
            raise ValueError(
                "The Chroma collection was built with a different embedding model "
                f"({stored_model!r}). Reindex with that model or use a new persist directory."
            )

    @property
    def count(self) -> int:
        return self._collection.count()

    def replace_document(
        self,
        source_file: str,
        chunks: Sequence[TextChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Replace all stored chunks for one file, preventing stale duplicates."""

        if len(chunks) != len(embeddings):
            raise ValueError("Chunk and embedding counts do not match")

        self._collection.delete(where={"source_file": source_file})
        if not chunks:
            return

        self._collection.upsert(
            ids=[_chunk_id(chunk) for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[chunk.metadata() for chunk in chunks],
            embeddings=[list(vector) for vector in embeddings],
        )

    def search(self, query_embedding: Sequence[float], top_k: int) -> list[SearchResult]:
        """Return nearest chunks ordered by cosine distance."""

        count = self.count
        if count == 0:
            return []
        response: dict[str, Any] = self._collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        documents = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]
        return [
            SearchResult(distance=float(distance), text=document, metadata=metadata or {})
            for document, metadata, distance in zip(documents, metadatas, distances, strict=True)
        ]


def _chunk_id(chunk: TextChunk) -> str:
    payload = (
        f"{chunk.source_file}\0{chunk.page}\0{chunk.chunk_index}\0{chunk.text}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
