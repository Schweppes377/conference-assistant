"""Speech recognition based on faster-whisper."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from faster_whisper import WhisperModel


@dataclass(frozen=True, slots=True)
class ASRSegment:
    """A timestamped text segment returned by the ASR model."""

    start: float
    end: float
    text: str


class ASRError(RuntimeError):
    """Raised when the Whisper model cannot transcribe audio."""


@lru_cache(maxsize=4)
def _load_model(model_name: str, device: str, compute_type: str) -> WhisperModel:
    """Load and cache one model instance per runtime configuration."""

    return WhisperModel(model_name, device=device, compute_type=compute_type)


class WhisperTranscriber:
    """Reusable faster-whisper transcriber with lazy, cached model loading."""

    def __init__(
        self,
        model_name: str = "medium",
        language: str | None = "ru",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.language = language
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_type = "float16" if self.device == "cuda" else "int8"

    @property
    def model(self) -> WhisperModel:
        """Return the cached model, loading it on first use."""

        try:
            return _load_model(self.model_name, self.device, self.compute_type)
        except Exception as exc:
            raise ASRError(
                f"Не удалось загрузить модель faster-whisper '{self.model_name}'."
            ) from exc

    def transcribe(self, audio: np.ndarray | str | Path) -> list[ASRSegment]:
        """Transcribe decoded mono audio or an audio file into timestamped segments."""

        try:
            segments, _ = self.model.transcribe(
                audio,
                language=self.language,
                beam_size=5,
                vad_filter=True,
            )
            return [
                ASRSegment(
                    start=float(segment.start),
                    end=float(segment.end),
                    text=segment.text.strip(),
                )
                for segment in segments
                if segment.text.strip()
            ]
        except ASRError:
            raise
        except Exception as exc:
            raise ASRError("Не удалось распознать аудио с помощью faster-whisper.") from exc
