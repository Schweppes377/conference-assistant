"""LLM-assisted detection of normative issues in transcript windows."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any

from app.analysis_models import (
    DetectedIssue,
    DetectionMetrics,
    DetectionResult,
)
from app.transcript import TranscriptSegment
from llm.base import LLMClient, LLMCompletion


DETECTION_SYSTEM_PROMPT = """Ты анализируешь стенограмму строительного совещания.

Найди только фрагменты, которые действительно требуют проверки по нормативным
документам: прямые нормативные вопросы, параметры проектных решений, возможные
несоответствия и сомнения о требованиях. Не включай обычные организационные
реплики, согласия, приветствия и поручения без нормативного содержания.

КРИТИЧЕСКИЕ ПРАВИЛА:
1. Анализируй соседние реплики совместно, в том числе разных участников.
2. Для одного обсуждаемого вопроса создай один результат, а не результат на
   каждую реплику.
   Разные проверяемые объекты или параметры (например, число выходов, ширина
   лестницы, ширина коридора и высота проёма) оформляй отдельными вопросами,
   даже если они относятся к одной категории безопасности.
3. normalized_question должен быть самостоятельным поисковым вопросом и
   включать все известные из локального контекста параметры.
4. search_terms — от 1 до 6 коротких русскоязычных терминов или устойчивых
   словосочетаний из исходного контекста для дополнительного поиска. Сохраняй
   точные специальные названия объекта и параметра; не подменяй их более общими.
5. Не отвечай на вопросы и не придумывай нормативные требования.
6. segment_ids выбирай только из переданной стенограммы. Включай все реплики,
   необходимые для понимания вопроса, но не добавляй несвязанный контекст.
7. category — короткий идентификатор латиницей в snake_case. Можно использовать
   fire_safety, evacuation, accessibility, ventilation, noise, structural,
   utilities, other или более точную категорию.
8. Верни только JSON без Markdown и пояснений.
9. Все текстовые поля результата формируй исключительно на русском языке,
   независимо от языка внутренних рассуждений модели. Поле normalized_question
   ОБЯЗАТЕЛЬНО должно быть на русском языке. Не переводи русскую стенограмму на
   английский язык. Поле reason и любые пользовательские summary/explanation
   также пиши только по-русски. Технические segment_ids и category оставляй в
   указанном машинном формате.
10. Не создавай новый вопрос из итогового перечисления уже обсуждённых тем или
    поручения проверить их. Используй исходные подробные реплики, где назван
    конкретный объект и параметр.

Формат:
{"issues":[{"segment_ids":["SEGMENT_0001"],"normalized_question":"...",
"search_terms":["точное название объекта","проверяемый параметр"],
"category":"evacuation","reason":"Почему фрагмент требует проверки"}]}

Если нормативно значимых фрагментов нет, верни {"issues":[]}."""

_RUSSIAN_RETRY_INSTRUCTION = """КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ ФОРМАТА:
Верни результат на русском языке. Все normalized_question, reason и любые другие
пользовательские текстовые поля, включая search_terms, должны быть исключительно
на русском языке.
Не переводи русскую стенограмму на английский. Верни полный исправленный JSON."""

_TRACEABILITY_RETRY_INSTRUCTION = """КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ ТРАССИРУЕМОСТИ:
Верни полный исправленный JSON. Для каждого вопроса укажи только те segment_ids,
в которых действительно обсуждаются этот объект и параметр. Все числовые
значения из normalized_question должны присутствовать в выбранных репликах.
Не присоединяй соседние реплики о другом объекте только из-за общей категории."""

_JSON_RETRY_INSTRUCTION = """КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ ФОРМАТА:
Предыдущий ответ не является корректным полным JSON. Верни заново только один
валидный JSON-объект формата {"issues":[...]}, без Markdown и пояснений."""

_CATEGORY = re.compile(r"^[a-z][a-z0-9_]{0,49}$")
_WORD = re.compile(r"[\wа-яё]+", re.IGNORECASE)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_STANDALONE_NUMBER = re.compile(r"(?<![\w])\d+(?:[.,]\d+)?(?![\w])")
_MEASUREMENT_UNIT = re.compile(
    r"\b(?:метр(?:а|ов)?|сантиметр(?:а|ов)?|миллиметр(?:а|ов)?|"
    r"квадратн\w*|процент\w*|децибел\w*|дб)\b",
    re.IGNORECASE,
)
_NUMBER_WORD = re.compile(
    r"\b(?:примерно|около|один|одна|одно|два|две|три|четыре|пять|"
    r"десять|двадцать|двадцати|двадцатью|пятьдесят|восемьдесят|"
    r"девяносто|сто|половин\w*)\b",
    re.IGNORECASE,
)
_IN_LIGHT = re.compile(r"\bв\s+свету\b", re.IGNORECASE)
_GENERIC_DEDUP_PREFIXES = (
    "безопас",
    "допуст",
    "документ",
    "эвакуац",
    "здани",
    "значен",
    "жил",
    "квартир",
    "многок",
    "миним",
    "нормат",
    "парамет",
    "площад",
    "провер",
    "проект",
    "соответ",
    "требов",
    "высот",
    "ширин",
)
_RECAP_CUE = re.compile(
    r"\b(?:подвед(?:е|ё)м\s+итог|подводя\s+итог|нам\s+нужно\s+проверить|"
    r"нужно\s+проверить\s+количество)\b",
    re.IGNORECASE,
)
_ENRICHED_CONTEXT = re.compile(
    r"Уточняющий контекст из совещания:\s*(.+?)"
    r"(?:\.\s*Соответствует ли|$)",
    re.IGNORECASE,
)
_CYRILLIC_LETTER = re.compile(r"[А-Яа-яЁё]")
_LATIN_LETTER = re.compile(r"[A-Za-z]")


class StructuredDetectionError(ValueError):
    """Raised when the detector returns invalid structured output."""


class TraceabilityDetectionError(StructuredDetectionError):
    """Raised when an issue is not supported by its selected transcript lines."""


@dataclass(frozen=True, slots=True)
class IssueDetectionConfig:
    """Windowing and generation settings for issue detection."""

    window_segments: int = 60
    overlap_segments: int = 8
    temperature: float = 0.1
    max_output_tokens: int = 4_000
    duplicate_similarity: float = 0.62
    duplicate_max_gap_seconds: float = 90.0

    def __post_init__(self) -> None:
        if self.window_segments < 2:
            raise ValueError("window_segments must be at least 2")
        if self.overlap_segments < 0 or self.overlap_segments >= self.window_segments:
            raise ValueError("overlap_segments must be smaller than window_segments")
        if not 0.0 < self.temperature <= 2.0:
            raise ValueError("temperature must be greater than 0 and at most 2")
        if self.max_output_tokens < 100:
            raise ValueError("max_output_tokens must be at least 100")
        if not 0.0 <= self.duplicate_similarity <= 1.0:
            raise ValueError("duplicate_similarity must be between 0 and 1")
        if self.duplicate_max_gap_seconds < 0:
            raise ValueError("duplicate_max_gap_seconds cannot be negative")


class NormativeIssueDetector:
    """Detect, validate, and deduplicate issues across transcript windows."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        config: IssueDetectionConfig | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.config = config or IssueDetectionConfig()

    def detect(self, transcript: Sequence[TranscriptSegment]) -> DetectionResult:
        """Return validated issues whose traceability comes from the transcript."""

        if not transcript:
            return DetectionResult([], DetectionMetrics(windows=0, llm_seconds=0.0))

        segment_map = {
            _segment_id(index): segment
            for index, segment in enumerate(transcript, start=1)
        }
        detected: list[DetectedIssue] = []
        completions: list[LLMCompletion] = []
        llm_seconds = 0.0
        russian_transcript = _is_predominantly_russian(
            " ".join(segment.text for segment in transcript)
        )

        windows = list(_windows(transcript, self.config))
        for start_index, window in windows:
            allowed_ids = {
                _segment_id(index)
                for index in range(start_index + 1, start_index + len(window) + 1)
            }
            prompt = _build_window_prompt(window, start_index)
            completion, elapsed = self._complete(prompt)
            llm_seconds += elapsed
            completions.append(completion)
            try:
                payload = _parse_payload(completion.content)
            except StructuredDetectionError:
                retry_prompt = f"{prompt}\n\n{_JSON_RETRY_INSTRUCTION}"
                completion, elapsed = self._complete(retry_prompt)
                llm_seconds += elapsed
                completions.append(completion)
                payload = _parse_payload(completion.content)
            invalid_fields = _unexpected_english_fields(payload, russian_transcript)
            if invalid_fields:
                retry_prompt = f"{prompt}\n\n{_RUSSIAN_RETRY_INSTRUCTION}"
                completion, elapsed = self._complete(retry_prompt)
                llm_seconds += elapsed
                completions.append(completion)
                payload = _parse_payload(completion.content)
                invalid_fields = _unexpected_english_fields(
                    payload,
                    russian_transcript,
                )
                if invalid_fields:
                    fields = ", ".join(invalid_fields)
                    raise StructuredDetectionError(
                        "LLM detection response is not in Russian: " + fields
                    )
            try:
                window_issues = _validated_issues(
                    payload,
                    allowed_ids=allowed_ids,
                    segment_map=segment_map,
                )
            except TraceabilityDetectionError:
                retry_prompt = f"{prompt}\n\n{_TRACEABILITY_RETRY_INSTRUCTION}"
                completion, elapsed = self._complete(retry_prompt)
                llm_seconds += elapsed
                completions.append(completion)
                payload = _parse_payload(completion.content)
                invalid_fields = _unexpected_english_fields(
                    payload,
                    russian_transcript,
                )
                if invalid_fields:
                    fields = ", ".join(invalid_fields)
                    raise StructuredDetectionError(
                        "LLM detection response is not in Russian: " + fields
                    )
                window_issues = _validated_issues(
                    payload,
                    allowed_ids=allowed_ids,
                    segment_map=segment_map,
                )
            detected.extend(window_issues)

        issues = _deduplicate(detected, segment_map, self.config)
        issues = [replace(issue, id=f"ISSUE_{index:03d}") for index, issue in enumerate(issues, 1)]
        return DetectionResult(
            issues=issues,
            metrics=DetectionMetrics(
                windows=len(windows),
                llm_seconds=llm_seconds,
                prompt_tokens=_sum_tokens(completions, "prompt_tokens"),
                completion_tokens=_sum_tokens(completions, "completion_tokens"),
                total_tokens=_sum_tokens(completions, "total_tokens"),
            ),
        )

    def _complete(self, prompt: str) -> tuple[LLMCompletion, float]:
        started = perf_counter()
        completion = self.llm_client.complete(
            system_prompt=DETECTION_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_output_tokens,
        )
        return completion, perf_counter() - started


def _windows(
    transcript: Sequence[TranscriptSegment],
    config: IssueDetectionConfig,
) -> list[tuple[int, Sequence[TranscriptSegment]]]:
    step = config.window_segments - config.overlap_segments
    result = []
    for start in range(0, len(transcript), step):
        window = transcript[start : start + config.window_segments]
        if not window:
            break
        result.append((start, window))
        if start + config.window_segments >= len(transcript):
            break
    return result


def _build_window_prompt(
    window: Sequence[TranscriptSegment],
    start_index: int,
) -> str:
    lines = ["Стенограмма для анализа:"]
    for offset, segment in enumerate(window, start=1):
        identifier = _segment_id(start_index + offset)
        lines.append(
            f"[{identifier}] {segment.start:.3f}-{segment.end:.3f} "
            f"{segment.speaker}: {segment.text}"
        )
    lines.append("Верни JSON с дедуплицированными нормативно значимыми вопросами.")
    return "\n".join(lines)


def _parse_payload(content: str) -> dict[str, Any]:
    candidate = content.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end < start:
        raise StructuredDetectionError("LLM detection response contains no JSON object")
    try:
        payload = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError as exc:
        raise StructuredDetectionError("LLM detection response contains invalid JSON") from exc
    if not isinstance(payload, dict):
        raise StructuredDetectionError("LLM detection response must be a JSON object")
    issues = payload.get("issues")
    if not isinstance(issues, list):
        raise StructuredDetectionError("LLM detection response has no issues array")
    if len(issues) > 50:
        raise StructuredDetectionError("LLM detection response contains too many issues")
    return payload


def _validated_issues(
    payload: dict[str, Any],
    *,
    allowed_ids: set[str],
    segment_map: dict[str, TranscriptSegment],
) -> list[DetectedIssue]:
    result = []
    for raw in payload["issues"]:
        if not isinstance(raw, dict):
            raise StructuredDetectionError("Each detected issue must be an object")
        raw_ids = raw.get("segment_ids")
        if not isinstance(raw_ids, list) or not raw_ids:
            raise StructuredDetectionError("Detected issue has no segment_ids")
        if not all(isinstance(value, str) for value in raw_ids):
            raise StructuredDetectionError("Detected issue has invalid segment_ids")
        if len(raw_ids) > 20:
            raise TraceabilityDetectionError(
                "Detected issue references too many segment_ids"
            )
        segment_ids = list(
            dict.fromkeys(value for value in raw_ids if value in allowed_ids)
        )
        if not segment_ids:
            raise TraceabilityDetectionError(
                "Detected issue references no known segment_ids"
            )

        question = _required_text(raw, "normalized_question", minimum=10, maximum=1_500)
        search_terms = _validated_search_terms(raw, question)
        category = _required_text(raw, "category", minimum=2, maximum=50).casefold()
        reason = _required_text(raw, "reason", minimum=5, maximum=1_500)
        if not _CATEGORY.fullmatch(category):
            raise StructuredDetectionError("Detected issue category must use snake_case")
        unsupported_numbers = _unsupported_significant_numbers(
            question,
            " ".join(segment_map[value].text for value in segment_ids),
        )
        if unsupported_numbers:
            raise TraceabilityDetectionError(
                "Detected issue question contains values unsupported by its "
                "segment_ids: " + ", ".join(unsupported_numbers)
            )
        result.append(
            _issue_from_segments(
                segment_ids,
                segment_map,
                normalized_question=question,
                search_terms=search_terms,
                category=category,
                reason=reason,
            )
        )
    return result


def _unsupported_significant_numbers(question: str, source_text: str) -> list[str]:
    """Find substantial numeric values not traceable to selected transcript lines."""

    source_numbers = set(_STANDALONE_NUMBER.findall(source_text))
    unsupported: list[str] = []
    for token in _STANDALONE_NUMBER.findall(question):
        try:
            value = float(token.replace(",", "."))
        except ValueError:  # pragma: no cover - regex already constrains the token
            continue
        if value >= 2 and token not in source_numbers:
            unsupported.append(token)
    return list(dict.fromkeys(unsupported))


def _required_text(
    raw: dict[str, Any],
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> str:
    value = raw.get(name)
    if not isinstance(value, str):
        raise StructuredDetectionError(f"Detected issue has invalid {name}")
    compact = " ".join(value.split())
    if not minimum <= len(compact) <= maximum:
        raise StructuredDetectionError(f"Detected issue has invalid {name} length")
    return compact


def _validated_search_terms(
    raw: dict[str, Any],
    normalized_question: str,
) -> list[str]:
    values = raw.get("search_terms")
    if values is None:
        return [normalized_question]
    if not isinstance(values, list) or not 1 <= len(values) <= 6:
        raise StructuredDetectionError(
            "Detected issue search_terms must contain from 1 to 6 items"
        )
    result: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise StructuredDetectionError("Detected issue has invalid search_terms")
        compact = " ".join(value.split())
        if not 3 <= len(compact) <= 200:
            raise StructuredDetectionError("Detected issue has invalid search term length")
        result.append(compact)
    return list(dict.fromkeys(result))


def _is_predominantly_russian(text: str) -> bool:
    cyrillic = len(_CYRILLIC_LETTER.findall(text))
    latin = len(_LATIN_LETTER.findall(text))
    return cyrillic >= 20 and cyrillic > latin


def _unexpected_english_fields(
    payload: dict[str, Any],
    russian_transcript: bool,
) -> list[str]:
    if not russian_transcript:
        return []
    invalid: list[str] = []
    for index, raw in enumerate(payload.get("issues", []), start=1):
        if not isinstance(raw, dict):
            continue
        for field in ("normalized_question", "reason"):
            value = raw.get(field)
            if isinstance(value, str) and _is_fully_english(value):
                invalid.append(f"issues[{index}].{field}")
        search_terms = raw.get("search_terms")
        if isinstance(search_terms, list):
            for term_index, value in enumerate(search_terms, start=1):
                if isinstance(value, str) and _is_fully_english(value):
                    invalid.append(
                        f"issues[{index}].search_terms[{term_index}]"
                    )
    return invalid


def _is_fully_english(text: str) -> bool:
    return (
        len(_LATIN_LETTER.findall(text)) >= 10
        and not _CYRILLIC_LETTER.search(text)
    )


def _issue_from_segments(
    segment_ids: Sequence[str],
    segment_map: dict[str, TranscriptSegment],
    *,
    normalized_question: str,
    search_terms: Sequence[str],
    category: str,
    reason: str,
) -> DetectedIssue:
    ordered_ids = sorted(segment_ids, key=_segment_number)
    segments = [segment_map[identifier] for identifier in ordered_ids]
    speakers = list(dict.fromkeys(segment.speaker for segment in segments))
    original_text = "\n".join(
        f"{segment.speaker}: {segment.text}" for segment in segments
    )
    enriched_question = _enrich_with_missing_parameters(
        normalized_question,
        segments,
    )
    enriched_search_terms = _enrich_search_terms(
        search_terms,
        enriched_question,
        segments,
    )
    return DetectedIssue(
        id="",
        start=min(segment.start for segment in segments),
        end=max(segment.end for segment in segments),
        speakers=speakers,
        segment_ids=ordered_ids,
        original_text=original_text,
        normalized_question=enriched_question,
        search_terms=enriched_search_terms,
        category=category,
        reason=reason,
    )


def _enrich_with_missing_parameters(
    question: str,
    segments: Sequence[TranscriptSegment],
) -> str:
    """Preserve measured project values even if the LLM omits them."""

    question_numbers = set(_NUMBER.findall(question))
    additions: list[str] = []
    word_measurement_added = False
    for segment in segments:
        text = " ".join(segment.text.split())
        numbers = set(_NUMBER.findall(text))
        has_word_measurement = bool(
            _MEASUREMENT_UNIT.search(text) and _NUMBER_WORD.search(text)
        )
        missing_digits = bool(numbers - question_numbers)
        if missing_digits or (
            has_word_measurement
            and not word_measurement_added
            and _jaccard(text, question) < 0.55
        ):
            additions.append(text.rstrip(" .!?"))
            word_measurement_added = word_measurement_added or has_word_measurement
        if _IN_LIGHT.search(text) and "выхода в свету" not in question.casefold():
            additions.append("высота эвакуационного выхода в свету")
        if len(additions) >= 3:
            break
    if not additions:
        return question
    base = question.rstrip(" .!?")
    context = "; ".join(dict.fromkeys(additions))
    return (
        f"{base}. Уточняющий контекст из совещания: {context}. "
        "Соответствует ли это нормативным требованиям?"
    )


def _enrich_search_terms(
    search_terms: Sequence[str],
    normalized_question: str,
    segments: Sequence[TranscriptSegment],
) -> list[str]:
    """Preserve specific transcript phrases as additional semantic queries.

    The extraction is deliberately domain-agnostic: it keeps compact clauses
    from the cited transcript segments instead of mapping known objects to
    hand-written normative terminology.
    """

    result = list(dict.fromkeys(term.strip() for term in search_terms if term.strip()))
    match = _ENRICHED_CONTEXT.search(normalized_question)
    if match:
        for clause in re.split(r"\s*;\s*", match.group(1)):
            _append_search_clause(result, clause)
            if len(result) >= 6:
                break
    for segment in segments:
        for clause in re.split(r"[.!?;:]+", segment.text):
            _append_search_clause(result, clause)
            if len(result) >= 6:
                return result[:6]
    return result[:6]


def _append_search_clause(result: list[str], clause: str) -> None:
    """Append one useful compact clause unless it duplicates an LLM term."""

    compact = " ".join(clause.strip(" .!?").split())
    words = _WORD.findall(compact)
    if not 2 <= len(words) <= 12 or not 8 <= len(compact) <= 200:
        return
    if any(_jaccard(compact, existing) >= 0.8 for existing in result):
        return
    result.append(compact)


def _deduplicate(
    issues: Sequence[DetectedIssue],
    segment_map: dict[str, TranscriptSegment],
    config: IssueDetectionConfig,
) -> list[DetectedIssue]:
    merged: list[DetectedIssue] = []
    for candidate in sorted(issues, key=lambda item: (item.start, item.end)):
        duplicate_index = next(
            (
                index
                for index, existing in enumerate(merged)
                if _are_duplicates(existing, candidate, config)
            ),
            None,
        )
        if duplicate_index is None:
            merged.append(candidate)
            continue
        existing = merged[duplicate_index]
        combined_ids = list(
            dict.fromkeys([*existing.segment_ids, *candidate.segment_ids])
        )
        fuller = max(
            (existing, candidate),
            key=lambda item: len(item.normalized_question),
        )
        merged[duplicate_index] = _issue_from_segments(
            combined_ids,
            segment_map,
            normalized_question=fuller.normalized_question,
            search_terms=list(
                dict.fromkeys([*existing.search_terms, *candidate.search_terms])
            )[:6],
            category=fuller.category,
            reason=max((existing.reason, candidate.reason), key=len),
        )
    return _drop_recap_issues(merged)


def _drop_recap_issues(issues: Sequence[DetectedIssue]) -> list[DetectedIssue]:
    """Remove late recap items that duplicate several earlier detailed issues."""

    result: list[DetectedIssue] = []
    for candidate in issues:
        earlier = [item for item in result if item.end <= candidate.start]
        repeated_topics = sum(
            _share_specific_lexeme(item, candidate) for item in earlier
        )
        if _RECAP_CUE.search(candidate.original_text) and repeated_topics >= 2:
            continue
        result.append(candidate)
    return result


def _are_duplicates(
    first: DetectedIssue,
    second: DetectedIssue,
    config: IssueDetectionConfig,
) -> bool:
    gap = max(0.0, max(first.start, second.start) - min(first.end, second.end))
    if gap > config.duplicate_max_gap_seconds:
        return False
    first_ids = set(first.segment_ids)
    second_ids = set(second.segment_ids)
    id_overlap = bool(first_ids & second_ids)
    similarity = _jaccard(first.normalized_question, second.normalized_question)
    related_continuation = (
        first.category == second.category
        and gap <= 15.0
        and not _NUMBER.search(first.normalized_question)
        and not _NUMBER.search(second.normalized_question)
        and _share_specific_lexeme(first, second)
    )
    return (
        similarity >= config.duplicate_similarity
        or (id_overlap and similarity >= 0.50)
        or related_continuation
    )


def _share_specific_lexeme(
    first: DetectedIssue,
    second: DetectedIssue,
) -> bool:
    first_words = _dedup_lexemes(
        " ".join([first.normalized_question, *first.search_terms])
    )
    second_words = _dedup_lexemes(
        " ".join([second.normalized_question, *second.search_terms])
    )
    return bool(first_words & second_words)


def _dedup_lexemes(text: str) -> set[str]:
    result = set()
    for word in _WORD.findall(text.casefold().replace("ё", "е")):
        if (
            len(word) < 7
            or any(word.startswith(prefix) for prefix in _GENERIC_DEDUP_PREFIXES)
        ):
            continue
        result.add(word[:6])
    return result


def _jaccard(first: str, second: str) -> float:
    first_words = {word.casefold() for word in _WORD.findall(first) if len(word) > 2}
    second_words = {word.casefold() for word in _WORD.findall(second) if len(word) > 2}
    if not first_words or not second_words:
        return 0.0
    return len(first_words & second_words) / len(first_words | second_words)


def _segment_id(index: int) -> str:
    return f"SEGMENT_{index:04d}"


def _segment_number(identifier: str) -> int:
    return int(identifier.rsplit("_", 1)[1])


def _sum_tokens(completions: Sequence[LLMCompletion], name: str) -> int | None:
    values = [getattr(completion, name) for completion in completions]
    known = [value for value in values if isinstance(value, int)]
    return sum(known) if known else None
