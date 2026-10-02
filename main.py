"""Command-line entry point for the meeting transcription MVP."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv

from app import MeetingTranscriptionPipeline, TranscriptSegment
from asr import ASRError
from diarization import (
    DiarizationError,
    DiarizationModelError,
    MissingHFTokenError,
)
from utils.audio import AudioDecodingError


def _format_timestamp(seconds: float, *, round_up: bool = False) -> str:
    total_seconds = math.ceil(seconds) if round_up else math.floor(seconds)
    hours, remainder = divmod(max(total_seconds, 0), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _format_segment(segment: TranscriptSegment) -> str:
    start = _format_timestamp(segment.start)
    end = _format_timestamp(segment.end, round_up=True)
    return f"[{start} - {end}] {segment.speaker}: {segment.text}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Распознавание речи и определение говорящих в локальном аудиофайле."
    )
    parser.add_argument("audio", type=Path, help="Путь к локальному аудиофайлу")
    parser.add_argument(
        "--language",
        default="ru",
        help="Код языка Whisper; укажите auto для автоопределения (по умолчанию: ru)",
    )
    parser.add_argument(
        "--model",
        default="medium",
        help="Модель faster-whisper (по умолчанию: medium)",
    )
    return parser


def main() -> int:
    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env")
    args = _build_parser().parse_args()

    if not args.audio.is_file():
        print(f"Ошибка: аудиофайл не найден: {args.audio}", file=sys.stderr)
        return 2

    if not torch.cuda.is_available():
        print(
            "Предупреждение: CUDA недоступна; используется CPU fallback. "
            "Обработка будет значительно медленнее.",
            file=sys.stderr,
        )

    language = None if args.language.lower() == "auto" else args.language

    try:
        pipeline = MeetingTranscriptionPipeline(
            model_name=args.model,
            language=language,
        )
        transcript = pipeline.process(args.audio)
    except (
        AudioDecodingError,
        MissingHFTokenError,
        DiarizationModelError,
        DiarizationError,
        ASRError,
    ) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1

    if not transcript:
        print("Речь в аудиофайле не обнаружена.")
        return 0

    for segment in transcript:
        print(_format_segment(segment))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
