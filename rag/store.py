"""Persistent ChromaDB storage with deterministic, idempotent updates."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from math import log
from rag.query import tokens

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
        self._corpus_cache = None

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
        self._corpus_cache = None
        if not chunks:
            return

        self._collection.upsert(
            ids=[_chunk_id(chunk) for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[chunk.metadata() for chunk in chunks],
            embeddings=[list(vector) for vector in embeddings],
        )
        self._corpus_cache = None

    def corpus(self) -> tuple[list[SearchResult], np.ndarray, list[set[str]], dict[str, float]]:
        """Read existing chunks/vectors once; create no secondary disk index."""
        if self._corpus_cache is None:
            raw = self._collection.get(include=['documents','metadatas','embeddings'])
            records = [SearchResult(1.0, text, metadata or {}) for text, metadata
                       in zip(raw['documents'], raw['metadatas'], strict=True)]
            matrix = np.asarray(raw['embeddings'], dtype=np.float32)
            term_sets = [tokens(r.text) for r in records]
            frequencies: dict[str, int] = {}
            for terms in term_sets:
                for term in terms: frequencies[term] = frequencies.get(term,0)+1
            idf = {t: log(1+(len(records)+1)/(count+1)) for t,count in frequencies.items()}
            self._corpus_cache = records,matrix,term_sets,idf
        return self._corpus_cache

    def lexical_search(self, query: str, query_embedding: Sequence[float], top_k: int) -> list[SearchResult]:
        """Lexical candidates retain genuine cosine scores from stored vectors."""
        records,matrix,term_sets,idf = self.corpus()
        terms = tokens(query)
        if not records or not terms: return []
        rows = [(sum(idf.get(t,0) for t in terms & ts), index) for index,ts in enumerate(term_sets)]
        rows = sorted((row for row in rows if row[0]>0), key=lambda r:(-r[0],r[1]))[:top_k]
        vector = np.asarray(query_embedding,dtype=np.float32)
        result = []
        for rank,(lexical,index) in enumerate(rows,1):
            stored = matrix[index]
            denom = float(np.linalg.norm(stored)*np.linalg.norm(vector))
            cosine = float(np.dot(stored,vector)/denom) if denom else 0.0
            record=records[index]
            result.append(SearchResult(1.-cosine,record.text,{**record.metadata,
                '_retrieval_method':'lexical','_initial_rank':rank,'_lexical_score':lexical}))
        return result

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
