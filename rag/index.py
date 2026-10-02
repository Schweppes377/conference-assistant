"""CLI for building the local normative-document vector index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag.chunking import chunk_pages
from rag.config import RAGConfig
from rag.embeddings import SentenceTransformerEmbedder
from rag.models import SourceInfo
from rag.pdf_loader import (
    PDFReadError,
    discover_pdf_files,
    extract_pdf_pages,
    load_source_catalog,
)
from rag.store import VectorStore


def build_index(config: RAGConfig) -> int:
    """Index all readable PDFs and return the number of stored chunks."""

    catalog = load_source_catalog(config.documents_dir)
    pdf_files = discover_pdf_files(config.documents_dir)
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {config.documents_dir}")

    embedder = SentenceTransformerEmbedder(
        config.embedding_model,
        batch_size=config.embedding_batch_size,
    )
    store = VectorStore(config.persist_dir, config.collection_name, config.embedding_model)
    print(f"Embedding device: {embedder.device}")

    indexed_documents = 0
    for path in pdf_files:
        source = catalog.get(path.name) or _fallback_source(path)
        if not source.has_extractable_text:
            store.replace_document(path.name, [], [])
            print(f"SKIP {path.name}: no extractable text layer")
            continue

        try:
            pages = extract_pdf_pages(path, source)
        except PDFReadError as exc:
            print(f"SKIP {path.name}: {exc}", file=sys.stderr)
            continue
        if not pages:
            store.replace_document(path.name, [], [])
            print(f"SKIP {path.name}: PyMuPDF extracted no text")
            continue

        chunks = chunk_pages(
            pages,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        embeddings = embedder.encode_documents([chunk.text for chunk in chunks])
        store.replace_document(path.name, chunks, embeddings)
        indexed_documents += 1
        print(f"OK   {path.name}: {len(pages)} text pages, {len(chunks)} chunks")

    if indexed_documents == 0:
        raise RuntimeError("No documents with extractable text were indexed")
    print(f"Index ready: {store.count} chunks in {config.persist_dir}")
    return store.count


def _fallback_source(path: Path) -> SourceInfo:
    print(f"WARN {path.name}: no entry in sources.json; using file name as title")
    return SourceInfo(
        file=path.name,
        title=path.stem,
        designation="",
        source_url="",
        status="unknown",
        edition_year="",
        has_extractable_text=True,
    )


def _parse_args() -> argparse.Namespace:
    defaults = RAGConfig()
    parser = argparse.ArgumentParser(description="Index normative PDF documents in ChromaDB.")
    parser.add_argument("--documents-dir", type=Path, default=defaults.documents_dir)
    parser.add_argument("--persist-dir", type=Path, default=defaults.persist_dir)
    parser.add_argument("--collection", default=defaults.collection_name)
    parser.add_argument("--model", default=defaults.embedding_model)
    parser.add_argument("--chunk-size", type=int, default=defaults.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=defaults.chunk_overlap)
    parser.add_argument("--batch-size", type=int, default=defaults.embedding_batch_size)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        config = RAGConfig(
            documents_dir=args.documents_dir.resolve(),
            persist_dir=args.persist_dir.resolve(),
            collection_name=args.collection,
            embedding_model=args.model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            embedding_batch_size=args.batch_size,
        )
        build_index(config)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Indexing error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Indexing failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
