"""CLI for the offline audio → normative report pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv

from app.analysis_models import MeetingAnalysis
from app.issue_detection import (
    IssueDetectionConfig,
    NormativeIssueDetector,
    StructuredDetectionError,
)
from app.meeting_analysis import EmptyTranscriptError, MeetingAnalysisPipeline
from app.pipeline import MeetingTranscriptionPipeline
from app.reporting import save_meeting_analysis
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
    defaults = QAConfig.from_env()
    parser = argparse.ArgumentParser(
        description="Полный offline-анализ аудиозаписи совещания по нормативам."
    )
    parser.add_argument("audio", type=Path, help="Путь к локальному аудиофайлу")
    parser.add_argument("--language", default="ru", help="Язык Whisper или auto")
    parser.add_argument("--asr-model", default="medium", help="Модель faster-whisper")
    parser.add_argument("--llm-model", default=defaults.model, help="Groq model ID")
    parser.add_argument(
        "--top-k",
        type=int,
        default=defaults.top_k,
        help="Количество reranked-фрагментов в контексте LLM",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=defaults.candidate_k,
        help="Количество кандидатов после расширенного retrieval",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=defaults.max_context_chars,
    )
    parser.add_argument("--min-score", type=float, default=defaults.min_relevance_score)
    parser.add_argument("--window-segments", type=int, default=60)
    parser.add_argument("--window-overlap", type=int, default=8)
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
    if not args.audio.is_file():
        print(f"Ошибка: аудиофайл не найден: {args.audio}", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print(
            "Предупреждение: CUDA недоступна; ASR и диаризация используют CPU fallback.",
            file=sys.stderr,
        )

    language = None if args.language.casefold() == "auto" else args.language
    try:
        qa_defaults = QAConfig.from_env()
        qa_config = QAConfig(
            model=args.llm_model,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            max_context_chars=args.max_context_chars,
            min_relevance_score=args.min_score,
            temperature=qa_defaults.temperature,
            max_output_tokens=qa_defaults.max_output_tokens,
            request_timeout_seconds=qa_defaults.request_timeout_seconds,
        )
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
            config=IssueDetectionConfig(
                window_segments=args.window_segments,
                overlap_segments=args.window_overlap,
            ),
        )
        pipeline = MeetingAnalysisPipeline(
            transcription_pipeline=MeetingTranscriptionPipeline(
                model_name=args.asr_model,
                language=language,
            ),
            issue_detector=detector,
            qa_service=qa_service,
        )
        analysis = pipeline.analyze(args.audio)
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
    if args.debug:
        _print_debug(analysis)
    return 0


def _print_summary(analysis: MeetingAnalysis) -> None:
    stats = analysis.statistics
    print("Анализ совещания завершён.")
    print(f"Реплик: {stats.transcript_segments}")
    print(f"Спикеры: {', '.join(stats.speakers)}")
    print(f"Нормативных вопросов: {stats.detected_issues}")
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
