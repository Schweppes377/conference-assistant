"""Structured data exchanged between RAG pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class SourceInfo:
    """Catalog metadata for one normative document."""

    file: str
    title: str
    designation: str
    source_url: str
    status: str
    edition_year: str
    has_extractable_text: bool
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceInfo":
        return cls(
            file=str(data["file"]),
            title=str(data["title"]),
            designation=str(data.get("designation", "")),
            source_url=str(data.get("source_url", "")),
            status=str(data.get("status", "")),
            edition_year=str(data.get("edition_year", "")),
            has_extractable_text=bool(data.get("has_extractable_text", True)),
            notes=str(data.get("notes", "")),
        )


@dataclass(frozen=True, slots=True)
class PageText:
    """Text extracted from one one-based PDF page."""

    source_path: Path
    title: str
    designation: str
    page: int
    text: str


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A retrievable text fragment with source metadata."""

    source_file: str
    title: str
    designation: str
    page: int
    chunk_index: int
    text: str

    def metadata(self) -> dict[str, str | int]:
        return {
            "source_file": self.source_file,
            "title": self.title,
            "designation": self.designation,
            "page": self.page,
            "chunk_index": self.chunk_index,
        }


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One semantic search hit returned from ChromaDB."""

    distance: float
    text: str
    metadata: dict[str, Any]

    @property
    def score(self) -> float:
        return 1.0 - self.distance
