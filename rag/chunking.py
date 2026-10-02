"""Paragraph-aware chunking for extracted normative text."""

from __future__ import annotations

import re
from collections.abc import Iterable

from rag.models import PageText, TextChunk


_SENTENCE_BREAK = re.compile(r"(?<=[.!?;:])\s+")


def chunk_pages(
    pages: Iterable[PageText],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    """Split pages without cutting words and keep chunks inside page boundaries."""

    chunks: list[TextChunk] = []
    per_file_index: dict[str, int] = {}

    for page in pages:
        source_file = page.source_path.name
        for text in split_text(page.text, chunk_size, chunk_overlap):
            chunk_index = per_file_index.get(source_file, 0)
            chunks.append(
                TextChunk(
                    source_file=source_file,
                    title=page.title,
                    designation=page.designation,
                    page=page.page,
                    chunk_index=chunk_index,
                    text=text,
                )
            )
            per_file_index[source_file] = chunk_index + 1

    return chunks


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Create chunks near ``chunk_size`` while preserving paragraph boundaries."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    units: list[str] = []
    for paragraph in paragraphs:
        units.extend(_split_oversized(paragraph, chunk_size))

    result: list[str] = []
    current = ""
    for unit in units:
        candidate = unit if not current else f"{current}\n\n{unit}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            result.append(current)
        overlap = _whole_word_tail(current, chunk_overlap)
        candidate = f"{overlap}\n\n{unit}" if overlap else unit
        current = candidate if len(candidate) <= chunk_size else unit

    if current:
        result.append(current)
    return result


def _split_oversized(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]

    sentences = [part.strip() for part in _SENTENCE_BREAK.split(text) if part.strip()]
    units: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > limit:
            if current:
                units.append(current)
                current = ""
            units.extend(_split_words(sentence, limit))
            continue
        candidate = sentence if not current else f"{current} {sentence}"
        if len(candidate) <= limit:
            current = candidate
        else:
            units.append(current)
            current = sentence
    if current:
        units.append(current)
    return units


def _split_words(text: str, limit: int) -> list[str]:
    result: list[str] = []
    current: list[str] = []
    current_length = 0
    for word in text.split():
        added = len(word) + (1 if current else 0)
        if current and current_length + added > limit:
            result.append(" ".join(current))
            current = [word]
            current_length = len(word)
        else:
            current.append(word)
            current_length += added
    if current:
        result.append(" ".join(current))
    return result


def _whole_word_tail(text: str, limit: int) -> str:
    if not text or limit <= 0:
        return ""
    words = text.split()
    selected: list[str] = []
    length = 0
    for word in reversed(words):
        added = len(word) + (1 if selected else 0)
        if selected and length + added > limit:
            break
        if not selected and len(word) > limit:
            return ""
        selected.append(word)
        length += added
    return " ".join(reversed(selected))
