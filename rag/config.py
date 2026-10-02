"""Configuration for the local RAG index and retrieval commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class RAGConfig:
    """Runtime settings that can be overridden by the CLI."""

    documents_dir: Path = PROJECT_ROOT / "data" / "documents"
    persist_dir: Path = PROJECT_ROOT / "data" / "chroma"
    collection_name: str = "normative_documents"
    embedding_model: str = "intfloat/multilingual-e5-small"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    embedding_batch_size: int = 32

    def __post_init__(self) -> None:
        if self.chunk_size < 200:
            raise ValueError("chunk_size must be at least 200 characters")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.embedding_batch_size < 1:
            raise ValueError("embedding_batch_size must be positive")
