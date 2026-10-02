"""CLI for semantic retrieval from the local normative-document index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag.config import RAGConfig
from rag.retriever import IndexNotReadyError, SemanticRetriever
from utils.console import configure_utf8_console


def search_index(config: RAGConfig, query: str, top_k: int) -> int:
    """Run one semantic query and print readable ranked results."""

    if not query.strip():
        raise ValueError("Search query cannot be empty")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    retriever = SemanticRetriever(config)
    results = retriever.retrieve(query, top_k)
    print(f"Query: {query}")
    print(f"Embedding device: {retriever.device}; indexed chunks: {retriever.index_count}")

    for rank, result in enumerate(results, start=1):
        metadata = result.metadata
        title = metadata.get("title", "Unknown document")
        designation = metadata.get("designation") or "—"
        page = metadata.get("page", "—")
        print(
            f"\n[{rank}] score={result.score:.4f} distance={result.distance:.4f}\n"
            f"Document: {title}\n"
            f"Designation: {designation}\n"
            f"Page: {page}\n"
            f"Text: {result.text}"
        )
    return len(results)


def _parse_args() -> argparse.Namespace:
    defaults = RAGConfig()
    parser = argparse.ArgumentParser(description="Search the local normative-document index.")
    parser.add_argument("query", nargs="+", help="Semantic search query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--persist-dir", type=Path, default=defaults.persist_dir)
    parser.add_argument("--collection", default=defaults.collection_name)
    parser.add_argument("--model", default=defaults.embedding_model)
    parser.add_argument("--batch-size", type=int, default=defaults.embedding_batch_size)
    return parser.parse_args()


def main() -> int:
    configure_utf8_console()
    args = _parse_args()
    try:
        defaults = RAGConfig()
        config = RAGConfig(
            documents_dir=defaults.documents_dir,
            persist_dir=args.persist_dir.resolve(),
            collection_name=args.collection,
            embedding_model=args.model,
            chunk_size=defaults.chunk_size,
            chunk_overlap=defaults.chunk_overlap,
            embedding_batch_size=args.batch_size,
        )
        search_index(config, " ".join(args.query), args.top_k)
    except (FileNotFoundError, IndexNotReadyError, ValueError, RuntimeError) as exc:
        print(f"Search error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Search failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
