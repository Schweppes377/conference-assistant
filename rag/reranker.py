"""Lightweight lexical reranking for normative-document candidates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import log1p
from typing import Sequence

from rag.models import SearchResult


_TOKEN = re.compile(r"[0-9a-zа-яё]+", re.IGNORECASE)
_SPACE = re.compile(r"\s+")
_DETACHED_PREFIX = re.compile(
    r"\b(вне|внутри|между)\s+([а-яё]{5,})\b",
    re.IGNORECASE,
)
_STOP_WORDS = {
    "а",
    "без",
    "в",
    "для",
    "до",
    "и",
    "из",
    "или",
    "как",
    "какие",
    "какой",
    "ли",
    "на",
    "не",
    "по",
    "при",
    "с",
    "со",
    "что",
    "это",
}


@dataclass(frozen=True, slots=True)
class RerankedResult:
    """A semantic candidate enriched with transparent reranking signals."""

    result: SearchResult
    rerank_score: float
    matched_terms: tuple[str, ...]


def rerank_candidates(
    query: str,
    search_terms: Sequence[str],
    candidates: Sequence[SearchResult],
    *,
    top_k: int,
) -> list[RerankedResult]:
    """Prefer candidates matching specific multi-word terms from the question."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    term_pairs = list(
        dict.fromkeys(
            (term.strip(), normalized)
            for term in search_terms
            if (normalized := _normalize(term))
        )
    )
    query_tokens = _content_tokens(query)
    all_term_tokens: set[str] = set().union(
        *(_content_tokens(term) for _, term in term_pairs)
    ) if term_pairs else set()
    ranked: list[RerankedResult] = []
    for candidate in candidates:
        text = _normalize(candidate.text)
        text_tokens = set(_TOKEN.findall(text))
        matched_terms: list[str] = []
        coverages: list[float] = []
        specificities: list[float] = []
        strong_matches = 0
        for original, term in term_pairs:
            tokens = _content_tokens(term)
            if not tokens:
                continue
            coverage = _token_coverage(tokens, text_tokens)
            coverages.append(coverage)
            specificities.append(
                coverage * min(1.0, log1p(len(tokens)) / log1p(4))
            )
            if term in text:
                matched_terms.append(original.strip())
            elif coverage >= 0.75:
                matched_terms.append(original.strip())
            if coverage >= 0.75:
                strong_matches += 1

        term_coverage = max(coverages, default=0.0)
        specificity = max(specificities, default=0.0)
        breadth = (
            min(strong_matches, 3) / min(len(term_pairs), 3)
            if term_pairs
            else 0.0
        )
        query_overlap = _token_coverage(query_tokens, text_tokens)
        # Reward chunks that jointly cover several distinct details scattered
        # across the query-expansion terms (typically object, parameter and
        # building type). Using the union avoids letting one broad phrase win
        # merely because it has high semantic similarity.
        matched_detail_count = sum(
            any(_same_lexeme(token, candidate) for candidate in text_tokens)
            for token in all_term_tokens
        )
        detail_density = min(matched_detail_count, 8) / 8
        score = (
            0.35 * candidate.score
            + 0.15 * specificity
            + 0.07 * breadth
            + 0.03 * term_coverage
            + 0.05 * query_overlap
            + 0.35 * detail_density
        )
        ranked.append(
            RerankedResult(
                result=candidate,
                rerank_score=score,
                matched_terms=tuple(dict.fromkeys(matched_terms)),
            )
        )
    ranked.sort(
        key=lambda item: (item.rerank_score, item.result.score),
        reverse=True,
    )
    return ranked[:top_k]


def _normalize(value: str) -> str:
    normalized = value.casefold().replace("ё", "е")
    # ASR may split productive prefixes used in technical compound terms
    # (for example, "вне квартирные" instead of "внеквартирные").
    normalized = _DETACHED_PREFIX.sub(r"\1\2", normalized)
    return _SPACE.sub(" ", normalized).strip()


def _content_tokens(value: str) -> set[str]:
    return {
        token
        for token in _TOKEN.findall(_normalize(value))
        if len(token) > 1 and token not in _STOP_WORDS
    }


def _token_coverage(needles: set[str], haystack: set[str]) -> float:
    if not needles:
        return 0.0
    matched = sum(
        any(_same_lexeme(needle, candidate) for candidate in haystack)
        for needle in needles
    )
    return matched / len(needles)


def _same_lexeme(first: str, second: str) -> bool:
    if first == second:
        return True
    return len(first) >= 6 and len(second) >= 6 and first[:5] == second[:5]
