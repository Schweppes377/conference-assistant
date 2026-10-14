"""Resumable meeting analysis with one generation policy per operation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from app.analysis_models import (
    AnalyzedIssue, DetectedIssue, MeetingAnalysis, MeetingStatistics, PipelineTimings,
    STATUS_INSUFFICIENT_INFORMATION, STATUS_PARTIAL_INFORMATION, STATUS_PENDING,
    STATUS_POTENTIAL_NONCOMPLIANCE, STATUS_PROCESSING_ERROR, STATUS_REQUIREMENT_FOUND,
)
from app.issue_detection import NormativeIssueDetector, StructuredDetectionError, _windows
from app.progress import save_checkpoint, check_llm_identity
from app.transcript import TranscriptSegment, merge_transcript
from llm.base import LLMError
from llm.models import PreparedQuestion, SourceEvidence
from llm.runtime import RunLLMClient, pause_message
from llm.service import QuestionAnsweringService


class EmptyTranscriptError(RuntimeError):
    """Raised when no speech segments were recognized."""


_NONCOMPLIANCE_MARKERS = (
    "не соответствует", "не отвечает требован", "не выполняет требован",
    "меньше требуем", "меньше миним", "ниже миним", "превышает допустим",
    "следует увеличить", "требуется увеличить", "не удовлетворяет", "не соблюд",
)
_INSUFFICIENT_MARKERS = (
    "недостаточно информации", "в предоставленных фрагментах их значения не указаны",
    "в предоставленных фрагментах не указаны", "конкретные значения не указаны",
    "числовые значения не указаны", "числовые величины не указаны",
    "в приведённых нормативных фрагментах не указано",
    "невозможно определить по предоставленным", "источники не содержат",
)


class MeetingAnalysisPipeline:
    """Checkpoint transcript, detection windows, exact retrieval and responses."""

    def __init__(
        self, *, transcription_pipeline: Any = None,
        issue_detector: NormativeIssueDetector, qa_service: QuestionAnsweringService,
        audio_loader: Callable[..., Any] | None = None,
        checkpoint_path: Path | None = None, state: dict[str, Any] | None = None,
        max_llm_calls: int | None = None,
        on_stage: Callable[[int, str], None] | None = None,
    ) -> None:
        if max_llm_calls is not None and max_llm_calls < 1:
            raise ValueError("MAX_LLM_CALLS_PER_RUN must be positive")
        self.transcription_pipeline = transcription_pipeline
        self.issue_detector = issue_detector
        self.qa_service = qa_service
        self.audio_loader = audio_loader
        self.checkpoint_path = checkpoint_path
        self.state = state if state is not None else {}
        check_llm_identity(self.state, provider=qa_service.llm_client.provider,
                           model=qa_service.llm_client.model)
        self.explicit_budget = max_llm_calls
        self.on_stage = on_stage or (lambda number, description: None)
        self.runtime = RunLLMClient(
            qa_service.llm_client, state=self.state.setdefault("llm_runtime", {}),
            max_calls=max_llm_calls or 1, on_change=self._save,
        )
        self.qa_service.llm_client = self.runtime
        self.issue_detector.llm_client = self.runtime

    def _save(self) -> None:
        self.state["pending_issue_ids"] = [
            issue["id"] for issue in self.state.get("detected_issues", [])
            if issue["id"] not in self.state.get("answers", {})
        ]
        if self.checkpoint_path:
            save_checkpoint(self.checkpoint_path, self.state)

    def analyze(self, audio_path: str | Path) -> MeetingAnalysis:
        """Preserve audio entry point; text/resume modes never invoke it."""
        from utils.audio import SAMPLE_RATE, load_audio
        path = Path(audio_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Аудиофайл не найден: {path}")
        if self.transcription_pipeline is None:
            raise ValueError("Audio pipeline is not configured")
        timings = {}
        started = perf_counter()
        audio = (self.audio_loader or load_audio)(path, sample_rate=SAMPLE_RATE)
        timings["audio_decode_seconds"] = perf_counter() - started
        self.state["audio_duration_sec"] = len(audio) / SAMPLE_RATE
        self.on_stage(1, "Распознавание речи")
        started = perf_counter()
        asr = self.transcription_pipeline.transcriber.transcribe(audio)
        timings["asr_seconds"] = perf_counter() - started
        self.on_stage(2, "Определение говорящих")
        started = perf_counter()
        diarization = self.transcription_pipeline.diarizer.diarize(audio, SAMPLE_RATE)
        timings["diarization_seconds"] = perf_counter() - started
        self.state["diarization_segments"] = [asdict(s) for s in diarization]
        self.on_stage(3, "Формирование стенограммы")
        self.state["audio_timings"] = timings
        return self.analyze_transcript(merge_transcript(asr, diarization), audio_file=str(path))

    def analyze_transcript(
        self, transcript: list[TranscriptSegment], *, audio_file: str = "transcript",
    ) -> MeetingAnalysis:
        """Continue from transcript or saved issues without replaying paid work."""
        if not transcript:
            raise EmptyTranscriptError("Речь в стенограмме не обнаружена.")
        started = perf_counter()
        self.state.update({
            "checkpoint_version": 1, "transcript": [asdict(s) for s in transcript],
            "audio_file": audio_file, "run_status": "running", "stop_reason": None,
        })
        self.state.setdefault("created_at", datetime.now().astimezone().isoformat(timespec="seconds"))
        self.state.setdefault("answers", {})
        self.state.setdefault("prepared", {})
        self.state.setdefault("detection_progress", {})
        self.state.setdefault("detected_issues", [])
        self.state.setdefault("detection_complete", False)
        settings = {
            "qa": asdict(self.qa_service.config),
            "detection": asdict(self.issue_detector.config),
        }
        saved_settings = self.state.get("settings")
        if saved_settings is not None and saved_settings != settings:
            raise ValueError("Настройки checkpoint отличаются: используйте исходные параметры --resume.")
        self.state["settings"] = settings
        window_count = len(_windows(transcript, self.issue_detector.config))
        self.state["detection_windows"] = window_count
        self.runtime.max_calls = self.explicit_budget or max(
            2, 2 * (window_count + len(self.state["detected_issues"])),
        )
        self._save()

        try:
            self.on_stage(4, "Выделение нормативных вопросов" + (" (сохранено)" if self.state["detection_complete"] else ""))
            if not self.state["detection_complete"]:
                detection = self.issue_detector.detect(
                    transcript, progress=self.state["detection_progress"], on_progress=self._save,
                )
                self.state["detected_issues"] = [asdict(issue) for issue in detection.issues]
                self.state["detection_complete"] = True
                self.state["detection_seconds"] = detection.metrics.llm_seconds
                self._save()
            issues = [DetectedIssue(**raw) for raw in self.state["detected_issues"]]
            self.on_stage(5, "Поиск нормативных документов и анализ")
            self.runtime.max_calls = self.explicit_budget or max(2, 2 * (window_count + len(issues)))
            for issue in issues:
                if issue.id in self.state["answers"]:
                    continue
                prepared = None
                try:
                    raw_prepared = self.state["prepared"].get(issue.id)
                    if raw_prepared is None:
                        prepared = self.qa_service.prepare(
                            issue.normalized_question, search_terms=issue.search_terms,
                        )
                        self.state["prepared"][issue.id] = asdict(prepared)
                        self._save()
                    else:
                        prepared = PreparedQuestion.from_dict(raw_prepared)
                    qa = self.qa_service.answer(
                        issue.normalized_question, search_terms=issue.search_terms,
                        prepared=prepared, operation_id=issue.id,
                    )
                    analyzed = AnalyzedIssue(
                        **asdict(issue), answer=qa.answer,
                        status=_answer_status(
                            insufficient=qa.insufficient_context, partial=qa.partial_context,
                            answer=qa.answer, has_sources=bool(qa.cited_sources),
                            validation=qa.validation,
                        ), cited_sources=qa.cited_sources, retrieved_candidates=qa.retrieved_candidates,
                        retrieval_seconds=qa.timings.retrieval_seconds,
                        llm_seconds=qa.timings.llm_seconds, total_seconds=qa.timings.total_seconds,
                        prompt_tokens=qa.prompt_tokens, completion_tokens=qa.completion_tokens,
                        total_tokens=qa.total_tokens,
                        validation=qa.validation,
                        retrieval_trace=qa.retrieval_trace,
                    )
                except LLMError:
                    # API failures/budget exhaustion pause ALL further LLM
                    # calls. There is no outer generation retry.
                    raise
                except Exception as exc:
                    raw_prepared = self.state["prepared"].get(issue.id)
                    # Do not re-parse corrupt metadata inside the error handler.
                    candidates = prepared.retrieved_candidates if prepared is not None else []
                    analyzed = AnalyzedIssue(
                        **asdict(issue), answer="Ответ не прошёл локальную проверку или обработку.",
                        status=STATUS_PROCESSING_ERROR, error=f"{type(exc).__name__}: {exc}",
                        retrieved_candidates=candidates,
                    )
                self.state["answers"][issue.id] = asdict(analyzed)
                self._save()
            self.state["run_status"] = "completed"
        except StructuredDetectionError as exc:
            self.state["run_status"] = "validation_error"
            self.state["stop_reason"] = (
                "Detection не прошёл локальную проверку после двух генераций. "
                "Автоматические повторы остановлены: " + str(exc)
            )
        except LLMError as exc:
            self.state["run_status"] = "paused"
            self.state["stop_reason"] = pause_message(exc)
            diagnostic = getattr(exc, "diagnostic", None)
            self.state["stop_diagnostic"] = asdict(diagnostic) if diagnostic else None
        finally:
            self.state["last_run_seconds"] = perf_counter() - started
            self._save()
        return self._snapshot(transcript)

    def _snapshot(self, transcript: list[TranscriptSegment]) -> MeetingAnalysis:
        issues = []
        for raw in self.state["detected_issues"]:
            answer = self.state["answers"].get(raw["id"])
            if answer is None:
                candidates = self.state["prepared"].get(raw["id"], {}).get("retrieved_candidates", [])
                issues.append(AnalyzedIssue(
                    **raw, answer="Ожидает продолжения обработки.", status=STATUS_PENDING,
                    retrieved_candidates=[SourceEvidence(**s) for s in candidates],
                ))
            else:
                names = {f.name for f in fields(AnalyzedIssue)}
                values = {key: value for key, value in answer.items() if key in names}
                for key in ("cited_sources", "retrieved_candidates"):
                    values[key] = [SourceEvidence(**s) for s in answer.get(key, [])]
                issues.append(AnalyzedIssue(**values))
        usage = self.runtime.metrics()
        historical = self.state.get("historical_usage")
        if historical:
            usage["historical_usage"] = {
                name: historical.get(name) for name in ("prompt_tokens", "completion_tokens", "total_tokens")
            }
            usage["usage_incomplete"] = True
        audio = self.state.get("audio_timings", {})
        statistics = MeetingStatistics(
            transcript_segments=len(transcript), speakers=sorted({s.speaker for s in transcript}),
            detected_issues=len(issues),
            requirement_found=_count_status(issues, STATUS_REQUIREMENT_FOUND),
            partial_information=_count_status(issues, STATUS_PARTIAL_INFORMATION),
            insufficient_information=_count_status(issues, STATUS_INSUFFICIENT_INFORMATION),
            potential_noncompliance=_count_status(issues, STATUS_POTENTIAL_NONCOMPLIANCE),
            processing_errors=_count_status(issues, STATUS_PROCESSING_ERROR),
            pending_issues=_count_status(issues, STATUS_PENDING),
            detection_windows=self.state["detection_windows"],
            timings=PipelineTimings(
                audio_decode_seconds=audio.get("audio_decode_seconds", 0.0),
                asr_seconds=audio.get("asr_seconds", 0.0),
                diarization_seconds=audio.get("diarization_seconds", 0.0),
                detection_llm_seconds=self.state.get("detection_seconds", 0.0),
                retrieval_seconds=sum(i.retrieval_seconds for i in issues),
                answer_llm_seconds=sum(i.llm_seconds for i in issues),
                total_seconds=self.state["last_run_seconds"] + sum(audio.values()),
            ), prompt_tokens=usage["prompt_tokens"], completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"], llm_usage=usage,
        )
        return MeetingAnalysis(
            audio_file=self.state["audio_file"], created_at=self.state["created_at"],
            transcript=transcript, detected_issues=issues, statistics=statistics,
            run_status=self.state["run_status"], stop_reason=self.state["stop_reason"],
            checkpoint_file=str(self.checkpoint_path) if self.checkpoint_path else None,
            llm_provider=self.runtime.provider, llm_model=self.runtime.model,
            audio_duration_sec=self.state.get("audio_duration_sec"),
            diarization_segments=self.state.get("diarization_segments"),
        )


def _answer_status(*, insufficient: bool, partial: bool, answer: str, has_sources: bool,
                   validation: dict[str, Any] | None = None) -> str:
    if validation is not None:
        # Never infer project compliance from words inside quoted regulations.
        if insufficient or not has_sources:
            return STATUS_INSUFFICIENT_INFORMATION
        if partial or validation.get("conclusion") != "requirement_found":
            return STATUS_PARTIAL_INFORMATION
        return STATUS_REQUIREMENT_FOUND
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
