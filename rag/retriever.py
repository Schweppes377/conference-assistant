"""Reusable semantic retrieval service for CLI and downstream RAG consumers."""

from __future__ import annotations

from rag.config import RAGConfig
from rag.embeddings import SentenceTransformerEmbedder
from rag.models import SearchResult
from rag.store import VectorStore


class IndexNotReadyError(RuntimeError):
    """Raised when the local ChromaDB index is missing or empty."""


class SemanticRetriever:
    """Query the existing sentence-transformers + ChromaDB index."""

    def __init__(self, config: RAGConfig) -> None:
        self.config = config
        self._embedder: SentenceTransformerEmbedder | None = None
        self._store: VectorStore | None = None

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
