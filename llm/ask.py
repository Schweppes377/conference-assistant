"""CLI entry point for Question → RAG → GigaChat → grounded answer."""

from __future__ import annotations

import argparse
import json
import os
import sys

from dotenv import load_dotenv

from llm.base import HTTPDiagnostic, LLMError
from llm.config import QAConfig
from llm.gigachat_client import GigaChatClient
from llm.models import CitationValidationError, QAResult, SourceMetadataError
from llm.service import QuestionAnsweringService
from llm.runtime import RunLLMClient, pause_message
from rag.config import PROJECT_ROOT, RAGConfig
from rag.retriever import IndexNotReadyError, SemanticRetriever
from utils.console import configure_utf8_console


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Answer a question using the local normative RAG index and GigaChat."
    )
    parser.add_argument("question", nargs="*", help="Question about normative documents")
    parser.add_argument("--model", help="GigaChat model ID; overrides GIGACHAT_MODEL")
    parser.add_argument("--top-k", type=int, help="Number of retrieval results")
    parser.add_argument("--candidate-k", type=int, help="Number of reranking candidates")
    parser.add_argument("--max-context-chars", type=int, help="Maximum serialized context size")
    parser.add_argument("--min-score", type=float, help="Minimum cosine similarity score")
    parser.add_argument("--temperature", type=float, help="LLM temperature")
    parser.add_argument("--max-output-tokens", type=int, help="Maximum completion tokens")
    parser.add_argument("--debug", action="store_true", help="Show retrieval and latency details")
    parser.add_argument("--json", action="store_true", help="Print the complete structured result")
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List GigaChat models available to the configured API key",
    )
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> QAConfig:
    defaults = QAConfig.from_env()
    return QAConfig(
        model=args.model or defaults.model,
        top_k=args.top_k if args.top_k is not None else defaults.top_k,
        candidate_k=(
            args.candidate_k
            if args.candidate_k is not None
            else defaults.candidate_k
        ),
        max_context_chars=(
            args.max_context_chars
            if args.max_context_chars is not None
            else defaults.max_context_chars
        ),
        min_relevance_score=(
            args.min_score if args.min_score is not None else defaults.min_relevance_score
        ),
        temperature=(
            args.temperature if args.temperature is not None else defaults.temperature
        ),
        max_output_tokens=(
            args.max_output_tokens
            if args.max_output_tokens is not None
            else defaults.max_output_tokens
        ),
        request_timeout_seconds=defaults.request_timeout_seconds,
    )


def _print_result(result: QAResult, *, debug: bool) -> None:
    print("Вопрос:")
    print(result.question)
    print("\nОтвет:")
    print(result.answer)
    print("\nИсточники:")
    if not result.cited_sources:
        print("Источники не использованы.")
    for index, source in enumerate(result.cited_sources, start=1):
        print(
            f"\n{index}. {source.document}\n"
            f"   Обозначение: {source.designation or '—'}\n"
            f"   Страница: {source.page}\n"
            f"   Chunk ID: {source.chunk_id}\n"
            f"   Score: {source.score:.4f}; distance: {source.distance:.4f}\n"
            f"   Фрагмент: {source.text}"
        )

    if debug:
        print("\nDebug:")
        print(f"Модель: {result.model}")
        print(
            f"CANDIDATE_K={result.candidate_k}; TOP_K={result.top_k}; "
            f"temperature={result.temperature}; "
            f"context={result.context_characters}/{result.max_context_chars} chars"
        )
        print(
            f"Latency: retrieval={result.timings.retrieval_seconds:.3f}s; "
            f"llm={result.timings.llm_seconds:.3f}s; "
            f"total={result.timings.total_seconds:.3f}s"
        )
        print(
            "Tokens: "
            f"prompt={result.prompt_tokens}; completion={result.completion_tokens}; "
            f"total={result.total_tokens}"
        )
        print("Кандидаты после reranking:")
        for source in result.retrieved_candidates:
            print(
                f"- [{source.source_id}] score={source.score:.4f} "
                f"rerank={source.rerank_score:.4f} "
                f"distance={source.distance:.4f} {source.chunk_id}"
            )


def _print_http_diagnostic(diagnostic: HTTPDiagnostic) -> None:
    """Print an allow-listed subset of a failed GigaChat HTTP response."""

    print("\nGigaChat HTTP debug:", file=sys.stderr)
    print(f"  HTTP status: {diagnostic.status}", file=sys.stderr)
    print(
        f"  Error message: {diagnostic.message or 'not provided by GigaChat'}",
        file=sys.stderr,
    )
    print(f"  Error type: {diagnostic.error_type or 'not provided'}", file=sys.stderr)
    error_code = (
        diagnostic.error_code if diagnostic.error_code is not None else "not provided"
    )
    print(f"  Error code: {error_code}", file=sys.stderr)
    print(f"  Model: {diagnostic.model}", file=sys.stderr)
    print(f"  Endpoint: {diagnostic.endpoint}", file=sys.stderr)


def main() -> int:
    configure_utf8_console()
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    args = _parse_args()
    runtime = None
    try:
        config = _config_from_args(args)
        client = GigaChatClient.from_env(
            model=config.model,
            timeout_seconds=config.request_timeout_seconds,
        )
        if args.list_models:
            for model in client.list_available_models():
                print(model)
            return 0

        question = " ".join(args.question).strip()
        if not question:
            raise ValueError("Provide a question or use --list-models")
        budget = int(os.getenv("MAX_LLM_CALLS_PER_RUN", "2"))
        runtime = RunLLMClient(client, max_calls=budget)
        service = QuestionAnsweringService(
            retriever=SemanticRetriever(RAGConfig()),
            llm_client=runtime,
            config=config,
        )
        result = service.answer(question)
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        else:
            _print_result(result, debug=args.debug)
        return 0
    except LLMError as exc:
        print(f"Ошибка: {pause_message(exc)}", file=sys.stderr)
        if args.debug and exc.diagnostic is not None:
            _print_http_diagnostic(exc.diagnostic)
        return 1
    except (
        CitationValidationError,
        IndexNotReadyError,
        SourceMetadataError,
        ValueError,
    ) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Неожиданная ошибка Question → RAG → LLM: {exc}", file=sys.stderr)
        return 1
    finally:
        if args.debug:
            print("LLM provider: GigaChat", file=sys.stderr)
            if 'config' in locals():
                print(f"LLM model: {config.model}", file=sys.stderr)
        if args.debug and runtime is not None:
            print("LLM usage: " + json.dumps(runtime.metrics(), ensure_ascii=False), file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
