"""Reusable semantic retrieval service for CLI and downstream RAG consumers."""

from __future__ import annotations

from rag.config import RAGConfig
from rag.embeddings import SentenceTransformerEmbedder
from rag.models import SearchResult
from rag.store import VectorStore
from rag.context import related_chunks
from rag.context import chunk_key
import numpy as np


class IndexNotReadyError(RuntimeError):
    """Raised when the local ChromaDB index is missing or empty."""


class SemanticRetriever:
    """Query the existing sentence-transformers + ChromaDB index."""

    def __init__(self, config: RAGConfig) -> None:
        self.config = config
        self._embedder: SentenceTransformerEmbedder | None = None
        self._store: VectorStore | None = None
        self._query_vectors: dict[str, list[float]] = {}

    @property
    def device(self) -> str:
        return self._get_embedder().device

    @property
    def index_count(self) -> int:
        return self._get_store().count

    def retrieve(self, query: str, top_k: int) -> list[SearchResult]:
        """Return TOP-K semantic matches without printing or LLM-specific logic."""

        if not query.strip():
            raise ValueError("Search query cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be positive")

        store = self._get_store()
        if store.count == 0:
            raise IndexNotReadyError("The index is empty. Run: python -m rag.index")
        query_embedding = self._get_embedder().encode_query(query)
        return store.search(query_embedding, top_k)

    def retrieve_candidates(self, query: str, top_k: int) -> list[SearchResult]:
        """Bounded semantic+lexical union, one existing-model embedding/query."""
        if not query.strip() or top_k < 1:
            raise ValueError('Нужен непустой запрос и положительный top_k')
        store = self._get_store()
        if store.count == 0:
            raise IndexNotReadyError('The index is empty. Run: python -m rag.index')
        vector = self._get_embedder().encode_query(query)
        self._query_vectors[query] = vector
        semantic = [SearchResult(r.distance,r.text,{**r.metadata,'_retrieval_method':'semantic','_initial_rank':rank})
                    for rank,r in enumerate(store.search(vector,top_k),1)]
        return semantic + store.lexical_search(query,vector,top_k)

    def related(self, parent: SearchResult, *, limit: int = 2) -> list[SearchResult]:
        """Follow bounded explicit same-document references in stored chunks."""
        records,matrix,_,_ = self._get_store().corpus()
        linked = related_chunks(parent,records,limit=limit)
        vector = self._query_vectors.get(parent.metadata.get('_retrieval_query',''))
        if vector is None: return linked
        by_key={chunk_key(r): index for index,r in enumerate(records)}
        q=np.asarray(vector,dtype=np.float32)
        return [SearchResult(1.-float(np.dot(matrix[by_key[chunk_key(r)]],q) /
               max(float(np.linalg.norm(matrix[by_key[chunk_key(r)]])*np.linalg.norm(q)),1e-12)),
               r.text,{**r.metadata,'_score_inherited':False}) for r in linked]

    def _get_embedder(self) -> SentenceTransformerEmbedder:
        if self._embedder is None:
            self._embedder = SentenceTransformerEmbedder(
                self.config.embedding_model,
                batch_size=self.config.embedding_batch_size,
            )
        return self._embedder

    def _get_store(self) -> VectorStore:
        if self._store is None:
            if not self.config.persist_dir.is_dir():
                raise IndexNotReadyError(
                    f"ChromaDB index not found: {self.config.persist_dir}. "
                    "Run: python -m rag.index"
                )
            self._store = VectorStore(
                self.config.persist_dir,
                self.config.collection_name,
                self.config.embedding_model,
            )
        return self._store
