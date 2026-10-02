"""Offline audio → transcript → issues → grounded answers orchestration."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

from app.analysis_models import (
    AnalyzedIssue,
    MeetingAnalysis,
    MeetingStatistics,
    PipelineTimings,
    STATUS_INSUFFICIENT_INFORMATION,
    STATUS_PARTIAL_INFORMATION,
    STATUS_POTENTIAL_NONCOMPLIANCE,
    STATUS_PROCESSING_ERROR,
    STATUS_REQUIREMENT_FOUND,
)
from app.issue_detection import NormativeIssueDetector
from app.pipeline import MeetingTranscriptionPipeline
from app.transcript import merge_transcript
from llm.service import QuestionAnsweringService
from llm.base import EmptyResponseError, RateLimitError
from llm.models import CitationValidationError
from utils.audio import SAMPLE_RATE, load_audio


class EmptyTranscriptError(RuntimeError):
    """Raised when no speech segments were recognized."""


_NONCOMPLIANCE_MARKERS = (
    "не соответствует",
    "не отвечает требован",
    "не выполняет требован",
    "меньше требуем",
    "меньше миним",
    "ниже миним",
    "превышает допустим",
    "следует увеличить",
    "требуется увеличить",
    "не удовлетворяет",
    "не соблюд",
)

_INSUFFICIENT_MARKERS = (
    "недостаточно информации",
    "в предоставленных фрагментах их значения не указаны",
    "в предоставленных фрагментах не указаны",
    "конкретные значения не указаны",
    "числовые значения не указаны",
    "числовые величины не указаны",
    "в приведённых нормативных фрагментах не указано",
    "невозможно определить по предоставленным",
    "источники не содержат",
)


class MeetingAnalysisPipeline:
    """Run the existing audio and grounded QA components as one offline pipeline."""

    def __init__(
        self,
        *,
        transcription_pipeline: MeetingTranscriptionPipeline,
        issue_detector: NormativeIssueDetector,
        qa_service: QuestionAnsweringService,
        audio_loader: Callable[..., Any] = load_audio,
        issue_retry_attempts: int = 5,
        max_retry_delay_seconds: float = 60.0,
        sleeper: Callable[[float], None] = sleep,
    ) -> None:
        self.transcription_pipeline = transcription_pipeline
        self.issue_detector = issue_detector
        self.qa_service = qa_service
        self.audio_loader = audio_loader
        self.issue_retry_attempts = issue_retry_attempts
        self.max_retry_delay_seconds = max_retry_delay_seconds
        self.sleeper = sleeper
        if issue_retry_attempts < 0:
            raise ValueError("issue_retry_attempts cannot be negative")
        if max_retry_delay_seconds < 0:
            raise ValueError("max_retry_delay_seconds cannot be negative")

    def analyze(self, audio_path: str | Path) -> MeetingAnalysis:
        """Analyze one meeting while isolating failures of individual issues."""

        path = Path(audio_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Аудиофайл не найден: {path}")

        total_started = perf_counter()
        decode_started = perf_counter()
        audio = self.audio_loader(path, sample_rate=SAMPLE_RATE)
        decode_seconds = perf_counter() - decode_started

        asr_started = perf_counter()
        asr_segments = self.transcription_pipeline.transcriber.transcribe(audio)
        asr_seconds = perf_counter() - asr_started

        diarization_started = perf_counter()
        diarization_segments = self.transcription_pipeline.diarizer.diarize(
            audio,
            SAMPLE_RATE,
        )
        diarization_seconds = perf_counter() - diarization_started

        transcript = merge_transcript(asr_segments, diarization_segments)
        if not transcript:
            raise EmptyTranscriptError("Речь в аудиофайле не обнаружена.")

        detection = self.issue_detector.detect(transcript)
        analyzed_issues: list[AnalyzedIssue] = []
        for issue in detection.issues:
            try:
                qa_result = self._answer_with_rate_limit_retry(
                    issue.normalized_question,
                    issue.search_terms,
                )
                status = _answer_status(
                    insufficient=qa_result.insufficient_context,
                    partial=qa_result.partial_context,
                    answer=qa_result.answer,
                    has_sources=bool(qa_result.cited_sources),
                )
                analyzed_issues.append(
                    AnalyzedIssue(
                        **_issue_fields(issue),
                        answer=qa_result.answer,
                        status=status,
                        cited_sources=qa_result.cited_sources,
                        retrieved_candidates=qa_result.retrieved_candidates,
                        retrieval_seconds=qa_result.timings.retrieval_seconds,
                        llm_seconds=qa_result.timings.llm_seconds,
                        total_seconds=qa_result.timings.total_seconds,
                        prompt_tokens=qa_result.prompt_tokens,
                        completion_tokens=qa_result.completion_tokens,
                        total_tokens=qa_result.total_tokens,
                    )
                )
            except Exception as exc:
                analyzed_issues.append(
                    AnalyzedIssue(
                        **_issue_fields(issue),
                        answer="Ответ не сформирован из-за ошибки обработки этого вопроса.",
                        status=STATUS_PROCESSING_ERROR,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )

        retrieval_seconds = sum(item.retrieval_seconds for item in analyzed_issues)
        answer_llm_seconds = sum(item.llm_seconds for item in analyzed_issues)
        prompt_tokens = _sum_optional(
            detection.metrics.prompt_tokens,
            *(item.prompt_tokens for item in analyzed_issues),
        )
        completion_tokens = _sum_optional(
            detection.metrics.completion_tokens,
            *(item.completion_tokens for item in analyzed_issues),
        )
        total_tokens = _sum_optional(
            detection.metrics.total_tokens,
            *(item.total_tokens for item in analyzed_issues),
        )
        speakers = sorted({segment.speaker for segment in transcript})
        statistics = MeetingStatistics(
            transcript_segments=len(transcript),
            speakers=speakers,
            detected_issues=len(analyzed_issues),
            requirement_found=_count_status(analyzed_issues, STATUS_REQUIREMENT_FOUND),
            partial_information=_count_status(
                analyzed_issues,
                STATUS_PARTIAL_INFORMATION,
            ),
            insufficient_information=_count_status(
                analyzed_issues,
                STATUS_INSUFFICIENT_INFORMATION,
            ),
            potential_noncompliance=_count_status(
                analyzed_issues,
                STATUS_POTENTIAL_NONCOMPLIANCE,
            ),
            processing_errors=_count_status(analyzed_issues, STATUS_PROCESSING_ERROR),
            detection_windows=detection.metrics.windows,
            timings=PipelineTimings(
                audio_decode_seconds=decode_seconds,
                asr_seconds=asr_seconds,
                diarization_seconds=diarization_seconds,
                detection_llm_seconds=detection.metrics.llm_seconds,
                retrieval_seconds=retrieval_seconds,
                answer_llm_seconds=answer_llm_seconds,
                total_seconds=perf_counter() - total_started,
            ),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        return MeetingAnalysis(
            audio_file=str(path),
            created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            transcript=transcript,
            detected_issues=analyzed_issues,
            statistics=statistics,
        )

    def _answer_with_rate_limit_retry(
        self,
        question: str,
        search_terms: list[str],
    ) -> Any:
        validation_failures = 0
        max_attempts = self.issue_retry_attempts + 2
        for attempt in range(max_attempts):
            try:
                return self.qa_service.answer(
                    question,
                    search_terms=search_terms,
                )
            except RateLimitError as exc:
                if attempt >= max_attempts - 1:
                    raise
                requested_delay = exc.retry_after_seconds or 1.0
                delay = min(
                    max(requested_delay, 1.0) * (2**attempt),
                    self.max_retry_delay_seconds,
                )
                self.sleeper(delay + 0.25)
            except (CitationValidationError, EmptyResponseError):
                if validation_failures >= 1:
                    raise
                validation_failures += 1
        raise RuntimeError("Rate-limit retry loop ended unexpectedly")  # pragma: no cover


def _issue_fields(issue: Any) -> dict[str, Any]:
    return {
        "id": issue.id,
        "start": issue.start,
        "end": issue.end,
        "speakers": issue.speakers,
        "segment_ids": issue.segment_ids,
        "original_text": issue.original_text,
        "normalized_question": issue.normalized_question,
        "search_terms": issue.search_terms,
        "category": issue.category,
        "reason": issue.reason,
    }


def _answer_status(
    *,
    insufficient: bool,
    partial: bool,
    answer: str,
    has_sources: bool,
) -> str:
    folded = answer.casefold()
    if insufficient or not has_sources:
        return STATUS_INSUFFICIENT_INFORMATION
    if partial:
        return STATUS_PARTIAL_INFORMATION
    if any(marker in folded for marker in _INSUFFICIENT_MARKERS):
        return STATUS_INSUFFICIENT_INFORMATION
    if any(marker in folded for marker in _NONCOMPLIANCE_MARKERS):
        return STATUS_POTENTIAL_NONCOMPLIANCE
    return STATUS_REQUIREMENT_FOUND


def _count_status(issues: list[AnalyzedIssue], status: str) -> int:
    return sum(issue.status == status for issue in issues)


def _sum_optional(*values: int | None) -> int | None:
    known = [value for value in values if isinstance(value, int)]
    return sum(known) if known else None
