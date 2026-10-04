"""CLI for the offline audio → normative report pipeline."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from app.analysis_models import MeetingAnalysis
from app.issue_detection import (
    IssueDetectionConfig,
    NormativeIssueDetector,
    StructuredDetectionError,
)
from app.meeting_analysis import EmptyTranscriptError, MeetingAnalysisPipeline
from app.progress import read_input
from app.reporting import save_meeting_analysis
from app.transcript import TranscriptSegment
from asr import ASRError
from diarization import (
    DiarizationError,
    DiarizationModelError,
    MissingHFTokenError,
)
from llm.base import HTTPDiagnostic, LLMError
from llm.config import QAConfig
from llm.groq_client import GroqClient
from llm.service import QuestionAnsweringService
from rag.config import PROJECT_ROOT, RAGConfig
from rag.retriever import IndexNotReadyError, SemanticRetriever
from utils.audio import AudioDecodingError
from utils.console import configure_utf8_console


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Полный offline-анализ аудиозаписи совещания по нормативам."
    )
    parser.add_argument("audio", type=Path, nargs="?", help="Путь к локальному аудиофайлу")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--transcript", type=Path, help="Готовая JSON-стенограмма; пропустить аудиоэтап")
    inputs.add_argument("--issues", type=Path, help="analysis.json; пропустить аудио и detection")
    inputs.add_argument("--resume", type=Path, help="Продолжить из *_progress.json")
    parser.add_argument("--reanswer", action="store_true",
                        help="С --issues: явно пересчитать все ответы в новом запуске")
    parser.add_argument("--checkpoint", type=Path, help="Путь нового progress.json (создаётся автоматически)")
    parser.add_argument("--max-llm-calls", type=int, default=os.getenv("MAX_LLM_CALLS_PER_RUN"),
                        help="Бюджет генераций; автоматически 2 × (окна detection + вопросы)")
    parser.add_argument("--language", default="ru", help="Язык Whisper или auto")
    parser.add_argument("--asr-model", default="medium", help="Модель faster-whisper")
    parser.add_argument("--llm-model", default=None, help="Groq model ID")
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Количество reranked-фрагментов в контексте LLM",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=None,
        help="Количество кандидатов после расширенного retrieval",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=None,
    )
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--window-segments", type=int, default=None)
    parser.add_argument("--window-overlap", type=int, default=None)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "results",
    )
    parser.add_argument("--debug", action="store_true")
    return parser


def main() -> int:
    configure_utf8_console()
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    args = _build_parser().parse_args()
    text_input = args.transcript or args.issues or args.resume
    if bool(args.audio) == bool(text_input):
        print("Укажите аудио ИЛИ один из --transcript, --issues, --resume.", file=sys.stderr)
        return 2
    if args.reanswer and not args.issues:
        print("--reanswer применяется только с --issues.", file=sys.stderr)
        return 2

    language = None if args.language.casefold() == "auto" else args.language
    try:
        state = read_input(text_input, mode=(
            "resume" if args.resume else "issues" if args.issues else "transcript"
        )) if text_input else {}
        if args.reanswer:
            state["answers"] = {}
            state["prepared"] = {}
        settings = state.get("settings", {})
        qa_defaults = QAConfig(**settings["qa"]) if settings else QAConfig.from_env()
        overrides = {
            "model": args.llm_model, "top_k": args.top_k, "candidate_k": args.candidate_k,
            "max_context_chars": args.max_context_chars, "min_relevance_score": args.min_score,
        }
        qa_config = replace(qa_defaults, **{k: v for k, v in overrides.items() if v is not None})
        detector_config = IssueDetectionConfig(**settings["detection"]) if settings else IssueDetectionConfig()
        detector_config = replace(detector_config, **{
            k: v for k, v in {"window_segments": args.window_segments,
                             "overlap_segments": args.window_overlap}.items() if v is not None
        })
        if args.resume and args.checkpoint:
            raise ValueError("--resume обновляет исходный checkpoint; не указывайте --checkpoint.")
        checkpoint = args.resume or args.checkpoint or (
            args.results_dir / f"{(args.audio or text_input).stem}_{datetime.now():%Y%m%d_%H%M%S_%f}_progress.json"
        )
        checkpoint = checkpoint.resolve()
        if not args.resume and checkpoint.exists():
            raise ValueError("Checkpoint уже существует. Используйте --resume или новый путь.")
        if not args.resume and checkpoint == (args.audio or text_input).resolve():
            raise ValueError("Checkpoint не может перезаписать исходный файл.")
        llm_client = GroqClient.from_env(
            model=qa_config.model,
            timeout_seconds=qa_config.request_timeout_seconds,
        )
        retriever = SemanticRetriever(RAGConfig())
        qa_service = QuestionAnsweringService(
            retriever=retriever,
            llm_client=llm_client,
            config=qa_config,
        )
        detector = NormativeIssueDetector(
            llm_client=llm_client,
            config=detector_config,
        )
        transcription_pipeline = None
        if args.audio:
            import torch
            from app.pipeline import MeetingTranscriptionPipeline
            if not args.audio.is_file():
                raise FileNotFoundError(f"Аудиофайл не найден: {args.audio}")
            if not torch.cuda.is_available():
                print("CUDA недоступна; аудиоэтап использует CPU fallback.", file=sys.stderr)
            transcription_pipeline = MeetingTranscriptionPipeline(model_name=args.asr_model, language=language)
        pipeline = MeetingAnalysisPipeline(
            transcription_pipeline=transcription_pipeline, issue_detector=detector,
            qa_service=qa_service, checkpoint_path=checkpoint, state=state,
            max_llm_calls=args.max_llm_calls,
        )
        analysis = pipeline.analyze(args.audio) if args.audio else pipeline.analyze_transcript(
            [TranscriptSegment(**s) for s in state["transcript"]],
            audio_file=state["audio_file"],
        )
        paths = save_meeting_analysis(analysis, args.results_dir)
    except LLMError as exc:
        print(f"Ошибка LLM: {exc}", file=sys.stderr)
        if args.debug and exc.diagnostic is not None:
            _print_http_diagnostic(exc.diagnostic)
        return 1
    except (
        ASRError,
        AudioDecodingError,
        DiarizationError,
        DiarizationModelError,
        EmptyTranscriptError,
        IndexNotReadyError,
        MissingHFTokenError,
        StructuredDetectionError,
        ValueError,
        OSError,
    ) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Неожиданная ошибка анализа совещания: {exc}", file=sys.stderr)
        return 1

    _print_summary(analysis)
    print("\nФайлы:")
    print(f"- Стенограмма: {paths.transcript}")
    print(f"- Анализ: {paths.analysis}")
    print(f"- Отчёт: {paths.report}")
    print(f"- Прогресс: {analysis.checkpoint_file}")
    if analysis.stop_reason:
        print(analysis.stop_reason, file=sys.stderr)
        diagnostic = pipeline.state.get("stop_diagnostic")
        if args.debug and diagnostic:
            _print_http_diagnostic(HTTPDiagnostic(**diagnostic))
        if analysis.run_status == "paused":
            print(f'.\\run-direct.ps1 -m app.analyze_meeting --resume "{analysis.checkpoint_file}" --debug')
    if args.debug:
        _print_debug(analysis)
    return 0 if analysis.run_status == "completed" else 3


def _print_summary(analysis: MeetingAnalysis) -> None:
    stats = analysis.statistics
    print("Анализ совещания завершён." if analysis.run_status == "completed" else "Обработка приостановлена; прогресс сохранён.")
    print(f"Реплик: {stats.transcript_segments}")
    print(f"Спикеры: {', '.join(stats.speakers)}")
    print(f"Нормативных вопросов: {stats.detected_issues}")
    print(f"Ожидают обработки: {stats.pending_issues}")
    for issue in analysis.detected_issues:
        print(f"- [{issue.id}] {issue.normalized_question} ({issue.status})")


def _print_debug(analysis: MeetingAnalysis) -> None:
    stats = analysis.statistics
    timings = stats.timings
    print("\nDebug:")
    print(f"ASR: {timings.asr_seconds:.3f} с")
    print(f"Diarization: {timings.diarization_seconds:.3f} с")
    print(
        f"Detection: {timings.detection_llm_seconds:.3f} с; "
        f"окон: {stats.detection_windows}"
    )
    print(f"Retrieval: {timings.retrieval_seconds:.3f} с")
    print(f"Answer LLM: {timings.answer_llm_seconds:.3f} с")
    print(f"Total: {timings.total_seconds:.3f} с")
    print(
        "Tokens: "
        f"prompt={stats.prompt_tokens}; completion={stats.completion_tokens}; "
        f"total={stats.total_tokens}"
    )
    usage = stats.llm_usage
    for key in ("llm_calls", "successful_llm_calls", "corrective_retries", "run_llm_calls",
                "max_llm_calls_per_run", "detection_tokens", "answer_tokens", "retry_tokens",
                "usage_incomplete", "historical_usage"):
        if key in usage:
            print(f"{key}: {usage[key]}")
    for issue in analysis.detected_issues:
        print(f"\n[{issue.id}] {issue.normalized_question}")
        print(
            f"  status={issue.status}; retrieval={issue.retrieval_seconds:.3f} с; "
            f"llm={issue.llm_seconds:.3f} с"
        )
        print(f"  search_terms={issue.search_terms}")
        for source in issue.retrieved_candidates:
            print(
                f"  - [{source.source_id}] score={source.score:.4f} "
                f"rerank={source.rerank_score:.4f} "
                f"distance={source.distance:.4f} {source.chunk_id}"
            )
        if issue.error:
            print(f"  error={issue.error}")


def _print_http_diagnostic(diagnostic: HTTPDiagnostic) -> None:
    print(f"  HTTP status: {diagnostic.status}", file=sys.stderr)
    print(
        f"  Error message: {diagnostic.message or 'not provided by Groq'}",
        file=sys.stderr,
    )
    print(f"  Error type: {diagnostic.error_type or 'not provided'}", file=sys.stderr)
    code = diagnostic.error_code if diagnostic.error_code is not None else "not provided"
    print(f"  Error code: {code}", file=sys.stderr)
    print(f"  Model: {diagnostic.model}", file=sys.stderr)
    print(f"  Endpoint: {diagnostic.endpoint}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
