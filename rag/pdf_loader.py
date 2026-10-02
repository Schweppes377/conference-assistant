"""PDF catalog loading and page-level text extraction with PyMuPDF."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pymupdf

from rag.models import PageText, SourceInfo


class PDFReadError(RuntimeError):
    """Raised when a PDF cannot be opened or parsed."""


def load_source_catalog(documents_dir: Path) -> dict[str, SourceInfo]:
    """Load ``sources.json`` and return records keyed by file name."""

    catalog_path = documents_dir / "sources.json"
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Source catalog not found: {catalog_path}")

    try:
        raw_records = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read source catalog {catalog_path}: {exc}") from exc

    if not isinstance(raw_records, list):
        raise ValueError(f"Source catalog must contain a JSON array: {catalog_path}")

    records = [SourceInfo.from_dict(record) for record in raw_records]
    return {record.file: record for record in records}


def discover_pdf_files(documents_dir: Path) -> list[Path]:
    """Return PDF files in stable name order."""

    if not documents_dir.is_dir():
        raise FileNotFoundError(f"Documents directory not found: {documents_dir}")
    return sorted(documents_dir.glob("*.pdf"), key=lambda path: path.name.casefold())


def extract_pdf_pages(path: Path, source: SourceInfo) -> list[PageText]:
    """Extract ordered text blocks from every page of one PDF."""

    try:
        document = pymupdf.open(path)
    except Exception as exc:
        raise PDFReadError(f"Cannot open PDF {path.name}: {exc}") from exc

    pages: list[PageText] = []
    try:
        if document.needs_pass:
            raise PDFReadError(f"Password-protected PDF is not supported: {path.name}")

        for page_number, page in enumerate(document, start=1):
            blocks = page.get_text("blocks", sort=True)
            paragraphs = [
                _normalize_block(block[4])
                for block in blocks
                if len(block) > 6 and block[6] == 0 and block[4].strip()
            ]
            text = "\n\n".join(paragraph for paragraph in paragraphs if paragraph)
            if text:
                pages.append(
                    PageText(
                        source_path=path,
                        title=source.title,
                        designation=source.designation,
                        page=page_number,
                        text=text,
                    )
                )
    except PDFReadError:
        raise
    except Exception as exc:
        raise PDFReadError(f"Cannot extract text from {path.name}: {exc}") from exc
    finally:
        document.close()

    return pages


def _normalize_block(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return text.strip()
