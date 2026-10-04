"""Grounded Question → retrieval → LLM orchestration."""

from __future__ import annotations

import re
from collections.abc import Sequence
from decimal import Decimal
from time import perf_counter
from typing import Protocol

from llm.base import EmptyResponseError, LLMClient
from llm.config import QAConfig
from llm.models import (
    CitationValidationError, ContextValidationError, PreparedQuestion,
    QAResult, QATimings, SourceEvidence,
)
from llm.runtime import generate
from llm.prompts import (
    INSUFFICIENT_CONTEXT_ANSWER,
    PARTIAL_CONTEXT_MARKER,
    SYSTEM_PROMPT,
    build_user_prompt,
)
from rag.models import SearchResult
from rag.reranker import rerank_candidates


_SOURCE_CITATION = re.compile(r"\[\s*(SOURCE_\d+)\s*\]")
_MEASURED_VALUE = re.compile(
    r"(?<![\w.,])(?P<value>\d+(?:[.,]\d+)?)\s*"
    r"(?P<unit>м²|м2|кв\.\s*м|квадратн(?:ый|ых|ого)\s+метр(?:а|ов)?|"
    r"сантиметр(?:а|ов)?|миллиметр(?:а|ов)?|метр(?:а|ов)?|см|мм|м|дб(?:а)?|%|процент(?:а|ов)?|"
    r"чел(?:овек(?:а|ов)?)?)(?![A-Za-zА-Яа-яЁё\d²³])",
    re.IGNORECASE,
)


class Retriever(Protocol):
    """Provider-independent retrieval contract used by the Q&A service."""

    def retrieve(self, query: str, top_k: int) -> list[SearchResult]: ...


class QuestionAnsweringService:
    """Combine existing semantic retrieval with a replaceable LLM provider."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        llm_client: LLMClient,
        config: QAConfig,
    ) -> None:
        self.retriever = retriever
        self.llm_client = llm_client
        self.config = config

    def answer(
        self,
        question: str,
        *,
        search_terms: Sequence[str] = (),
        prepared: PreparedQuestion | None = None,
        operation_id: str = "standalone",
    ) -> QAResult:
        """Answer one question using only verified retrieved chunks."""

        clean_question = question.strip()
        if not clean_question:
            raise ValueError("Question cannot be empty")
        trusted_search_terms = [
            term.strip()
            for term in search_terms
            if isinstance(term, str) and term.strip()
        ]
        validation_input = " ".join([clean_question, *trusted_search_terms])

        total_started = perf_counter()
        prepared = prepared or self.prepare(clean_question, search_terms=trusted_search_terms)
        if prepared.question != clean_question or prepared.validation_input != validation_input:
            raise ValueError("Saved retrieval does not match this question/search_terms")
        evidence = prepared.retrieved_candidates
        retrieval_seconds = prepared.retrieval_seconds
        context_sources = prepared.context_sources

        if not context_sources:
            total_seconds = perf_counter() - total_started
            return QAResult(
                question=clean_question,
                answer=INSUFFICIENT_CONTEXT_ANSWER,
                cited_sources=[],
                retrieved_candidates=evidence,
                model=self.llm_client.model,
                top_k=self.config.top_k,
                candidate_k=self.config.candidate_k,
                max_context_chars=self.config.max_context_chars,
                context_characters=0,
                temperature=self.config.temperature,
                insufficient_context=True,
                partial_context=False,
                timings=QATimings(
                    retrieval_seconds=retrieval_seconds,
                    llm_seconds=0.0,
                    total_seconds=total_seconds,
                ),
            )

        user_prompt = build_user_prompt(clean_question, context_sources)
        completions = []
        llm_seconds = 0.0
        current_prompt = user_prompt
        for attempt in range(2):
            llm_started = perf_counter()
            try:
                completion = generate(
                    self.llm_client, stage="answer", operation=operation_id,
                    attempt=attempt, system_prompt=SYSTEM_PROMPT,
                    user_prompt=current_prompt, temperature=self.config.temperature,
                    max_tokens=self.config.max_output_tokens,
                )
                llm_seconds += perf_counter() - llm_started
                completions.append(completion)
                if not completion.content.strip():
                    raise EmptyResponseError("LLM returned an empty answer")
                cited_sources, insufficient, partial = _validate_completion(
                    completion.content,
                    validation_input,
                    context_sources,
                )
                break
            except ContextValidationError:
                # A second generation cannot supply a missing normative value.
                # Reject the ungrounded answer and preserve its token usage.
                raise
            except (CitationValidationError, EmptyResponseError) as exc:
                if attempt == 1:
                    if isinstance(exc, EmptyResponseError):
                        raise CitationValidationError("LLM twice returned an empty answer") from exc
                    raise
                current_prompt = (
                    user_prompt
                    + "\n\nКРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ ОТВЕТА:\n"
                    + str(exc)
                    + "\nПерепиши ответ, используя только явно подтвержденные "
                    "утверждения и числа. Для каждого утверждения укажи [SOURCE_N]. "
                    "Если подтверждена только часть составного вопроса, используй "
                    "предписанную формулировку о частичной информации."
                )

        answer_text = completion.content
        if partial and PARTIAL_CONTEXT_MARKER.casefold() not in answer_text.casefold():
            answer_text = answer_text.rstrip() + "\n\n" + PARTIAL_CONTEXT_MARKER

        return QAResult(
            question=clean_question,
            answer=answer_text,
            cited_sources=cited_sources,
            retrieved_candidates=evidence,
            model=completion.model,
            top_k=self.config.top_k,
            candidate_k=self.config.candidate_k,
            max_context_chars=self.config.max_context_chars,
            context_characters=len(user_prompt),
            temperature=self.config.temperature,
            insufficient_context=insufficient,
            partial_context=partial,
            timings=QATimings(
                retrieval_seconds=retrieval_seconds,
                llm_seconds=llm_seconds,
                total_seconds=perf_counter() - total_started,
            ),
            prompt_tokens=_sum_tokens(completions, "prompt_tokens"),
            completion_tokens=_sum_tokens(completions, "completion_tokens"),
            total_tokens=_sum_tokens(completions, "total_tokens"),
        )

    def prepare(
        self, question: str, *, search_terms: Sequence[str] = (),
    ) -> PreparedQuestion:
        """Run unchanged retrieval once; persist before making an API call."""
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("Question cannot be empty")
        terms = [term.strip() for term in search_terms if isinstance(term, str) and term.strip()]
        started = perf_counter()
        evidence = self._retrieve_candidates(clean_question, terms)
        relevant = [source for source in evidence if source.score >= self.config.min_relevance_score]
        context_sources = self._fit_context(relevant[:self.config.top_k], clean_question)
        return PreparedQuestion(
            clean_question, " ".join([clean_question, *terms]), evidence,
            context_sources, perf_counter() - started,
        )

    def _retrieve_candidates(
        self,
        question: str,
        search_terms: Sequence[str],
    ) -> list[SourceEvidence]:
        terms = list(
            dict.fromkeys(
                term.strip()
                for term in search_terms
                if isinstance(term, str) and term.strip()
            )
        )
        queries = _compose_retrieval_queries(question, terms)
        unique: dict[tuple[object, ...], SearchResult] = {}
        for query in queries:
            # A specific object phrase may be less semantically similar than a
            # broad regulatory formulation and therefore sit below the first
            # ten hits. Keep the configured candidate depth for every query;
            # the union is deduplicated and reduced by the lightweight
            # reranker before anything is sent to the LLM.
            for result in self.retriever.retrieve(query, self.config.candidate_k):
                key = _result_key(result)
                existing = unique.get(key)
                if existing is None or result.score > existing.score:
                    unique[key] = result
        ranked = rerank_candidates(
            question,
            terms,
            list(unique.values()),
            top_k=self.config.candidate_k,
        )
        return [
            SourceEvidence.from_search_result(
                index,
                item.result,
                rerank_score=item.rerank_score,
                matched_terms=list(item.matched_terms),
            )
            for index, item in enumerate(ranked, start=1)
        ]
    def _fit_context(
        self,
        sources: Sequence[SourceEvidence],
        question: str,
    ) -> list[SourceEvidence]:
        selected: list[SourceEvidence] = []
        for source in sources:
            candidate = [*selected, source]
            size = len(build_user_prompt(question, candidate))
            if size <= self.config.max_context_chars:
                selected.append(source)
            else:
                break
        return selected


def _compose_retrieval_queries(
    question: str,
    search_terms: Sequence[str],
) -> list[str]:
    """Build bounded object+parameter expansions for first-stage retrieval.

    Normative wording often contains a specific object in one user phrase and
    the checked parameter in another. Pairing the two longest (usually most
    descriptive) terms with the remaining terms gives semantic retrieval that
    combined signal without introducing domain-specific synonyms or rules.
    """

    base = list(dict.fromkeys([question, *search_terms]))
    anchors = sorted(search_terms, key=len, reverse=True)[:2]
    expansions = [
        f"{anchor} {term}"
        for anchor in anchors
        for term in search_terms
        if anchor != term
    ][:8]
    return list(dict.fromkeys([*base, *expansions]))


def _resolve_citations(
    answer: str,
    available_sources: Sequence[SourceEvidence],
) -> list[SourceEvidence]:
    source_map = {source.source_id: source for source in available_sources}
    ordered_ids = dict.fromkeys(_SOURCE_CITATION.findall(answer))
    invalid_ids = [source_id for source_id in ordered_ids if source_id not in source_map]
    if invalid_ids:
        raise CitationValidationError(
            "LLM cited source IDs that were not provided: " + ", ".join(invalid_ids)
        )
    return [source_map[source_id] for source_id in ordered_ids if source_id in source_map]


def _result_key(result: SearchResult) -> tuple[object, ...]:
    metadata = result.metadata
    return (
        metadata.get("source_file"),
        metadata.get("page"),
        metadata.get("chunk_index"),
    )


def _validate_measured_values(
    answer: str,
    question: str,
    cited_sources: Sequence[SourceEvidence],
    context_sources: Sequence[SourceEvidence],
) -> None:
    without_citations = _SOURCE_CITATION.sub("", answer)
    claimed = _measurements(without_citations)
    if not claimed:
        return
    # Normalize each evidence text independently: joining unrelated chunks may
    # accidentally combine a trailing number with the next chunk's unit.
    allowed = _measurements(question)
    for source in cited_sources:
        allowed.update(_measurements(source.text))
    invented = sorted(claimed - allowed)
    if invented:
        guidance = []
        missing_context = False
        for measurement in invented:
            matching_ids = [
                source.source_id
                for source in context_sources
                if measurement in _measurements(source.text)
            ]
            formatted = f"{measurement[0]}={measurement[1]}"
            if matching_ids:
                guidance.append(
                    formatted + " подтверждается " + ", ".join(matching_ids)
                )
            else:
                missing_context = True
                guidance.append(formatted + " отсутствует во всем контексте")
        error_class = ContextValidationError if missing_context else CitationValidationError
        error = error_class(
            "LLM answer contains measured values absent from the question and "
            "cited chunks. Исправь ссылки или удали значения: "
            + "; ".join(guidance)
        )
        error.diagnostics = _measurement_diagnostics(answer, question, cited_sources)
        raise error


def _validate_completion(
    answer: str,
    question: str,
    context_sources: Sequence[SourceEvidence],
) -> tuple[list[SourceEvidence], bool, bool]:
    cited_sources = _resolve_citations(answer, context_sources)
    folded = answer.casefold()
    has_insufficient_phrase = INSUFFICIENT_CONTEXT_ANSWER.casefold() in folded
    partial = _looks_partial(answer, question, cited_sources)
    insufficient = has_insufficient_phrase and not partial
    if not insufficient and not cited_sources:
        raise CitationValidationError(
            "LLM answer contains no citations to the provided sources"
        )
    _validate_measured_values(
        answer,
        question,
        cited_sources,
        context_sources,
    )
    return cited_sources, insufficient, partial


def _looks_partial(
    answer: str,
    question: str,
    cited_sources: Sequence[SourceEvidence],
) -> bool:
    folded = answer.casefold()
    if PARTIAL_CONTEXT_MARKER.casefold() in folded:
        return True
    missing_detail = (
        (
            "не указаны" in folded
            or "отсутствуют" in folded
            or "не содерж" in folded
        )
        and any(
            marker in folded
            for marker in (
                "конкретные числовые",
                "дневн",
                "ночн",
                "детали",
                "дополнительн",
                "часть вопроса",
                "часть информации",
            )
        )
    )
    if missing_detail and cited_sources:
        return True
    insufficient_phrase = INSUFFICIENT_CONTEXT_ANSWER.casefold()
    if insufficient_phrase in folded and cited_sources:
        remainder = folded.replace(insufficient_phrase, "")
        remainder = _SOURCE_CITATION.sub("", remainder)
        return len(remainder.strip()) >= 80
    question_folded = question.casefold()
    if "дневн" in question_folded and "ночн" in question_folded:
        noise_values = {
            value
            for source in cited_sources
            for kind, value in _measurements(source.text)
            if kind == "noise_db"
        }
        if len(noise_values) < 2:
            return True
    return False


def _measurements(text: str) -> set[tuple[str, Decimal]]:
    """Canonical dimension/value pairs, with exact Decimal unit conversion."""
    return {(kind, value) for kind, value, _ in _measurement_matches(text)}


def _measurement_matches(text: str) -> list[tuple[str, Decimal, str]]:
    result = []
    for match in _MEASURED_VALUE.finditer(text):
        value = Decimal(match.group("value").replace(",", "."))
        unit = match.group("unit").casefold()
        if unit in {"м²", "м2"} or unit.startswith(("кв.", "квадратн")):
            kind = "area_m2"
        elif unit == "мм" or unit.startswith("миллиметр"):
            kind, value = "length_m", value / Decimal(1000)
        elif unit == "см" or unit.startswith("сантиметр"):
            kind, value = "length_m", value / Decimal(100)
        elif unit == "м" or unit.startswith("метр"):
            kind = "length_m"
        elif unit.startswith("дб"):
            kind = "noise_db"
        elif unit == "%" or unit.startswith("процент"):
            kind = "percent"
        else:
            kind = "people"
        result.append((kind, value, match.group(0)))
    result.extend(_width_table_matches(text))
    return result


def _width_table_matches(text: str) -> list[tuple[str, Decimal, str]]:
    """Read width cells only in explicit width(m)/slope tables flattened by PDF.

    Require a table heading, a dimensioned width header, an incline column,
    and a numeric width immediately followed by a slope ratio. Other bare
    numbers, document IDs, areas and the ratio components are not evidence.
    """
    result = []
    for table in re.finditer(r"\bТаблица\s+\d+(?:\.\d+)*", text, re.IGNORECASE):
        block = text[table.end():]
        stop = re.search(r"\bТаблица\s+\d|П\s*р\s*и\s*м\s*е\s*ч\s*а\s*н\s*и\s*я", block, re.IGNORECASE)
        block = block[:stop.start()] if stop else block
        header = re.search(r"ширина\s*,\s*м(?![\w²])\s+уклон\b", block, re.IGNORECASE)
        if not header:
            continue
        body = block[header.end():]
        # A cell may defer to another code; remove these references, preserving
        # offsets for debug evidence instead of mistaking an СП ID for a width.
        cleaned = re.sub(r"По\s+СП\s+\d+(?:\.\d+)*", lambda m: " " * len(m[0]), body, flags=re.IGNORECASE)
        for row in re.finditer(r"(?<![\w.,:])(?P<width>\d+(?:[.,]\d+)?)\s+"
                               r"\d+(?:[.,]\d+)?\s*:\s*\d+(?:[.,]\d+)?(?![\w.,])", cleaned):
            value = Decimal(row["width"].replace(",", "."))
            result.append(("length_m", value, "ширина, м: " + body[row.start():row.end()]))
    return result


def _measurement_diagnostics(answer: str, question: str,
                             cited_sources: Sequence[SourceEvidence]) -> list[dict]:
    """Local debug evidence for each claim; never treats uncited chunks as allowed."""
    units = {"length_m": "m", "area_m2": "m²", "noise_db": "dB", "percent": "%", "people": "people"}
    question_values = _measurement_matches(question)
    source_values = [(s.source_id, match) for s in cited_sources for match in _measurement_matches(s.text)]
    diagnostics = []
    for kind, value in sorted(_measurements(_SOURCE_CITATION.sub("", answer))):
        q_matches = [f"{raw} -> {normalized} {units[kind]}"
                     for dimension, normalized, raw in question_values if (dimension, normalized) == (kind, value)]
        s_matches = [{"source_id": source_id, "text": raw, "canonical": f"{normalized} {units[kind]}"}
                     for source_id, (dimension, normalized, raw) in source_values
                     if (dimension, normalized) == (kind, value)]
        diagnostics.append({"value": str(value), "dimension": kind,
                            "canonical": f"{value} {units[kind]}",
                            "question_matches": q_matches, "cited_source_matches": s_matches,
                            "rejected": not (q_matches or s_matches)})
    return diagnostics


def _sum_tokens(completions: Sequence[object], name: str) -> int | None:
    values = [getattr(completion, name, None) for completion in completions]
    known = [value for value in values if isinstance(value, int)]
    return sum(known) if known else None
