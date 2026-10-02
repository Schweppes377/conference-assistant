"""Local speaker diarization based on pyannote Community-1."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r"(?s).*torchcodec is not installed correctly.*",
        category=UserWarning,
        module=r"pyannote\.audio\.core\.io",
    )
    from pyannote.audio import Pipeline

PIPELINE_ID = "pyannote/speaker-diarization-community-1"


@dataclass(frozen=True, slots=True)
class DiarizationSegment:
    """A timestamped speaker turn."""

    start: float
    end: float
    speaker: str


class MissingHFTokenError(RuntimeError):
    """Raised when HF_TOKEN is not available in the environment."""


class DiarizationModelError(RuntimeError):
    """Raised when the gated pyannote model cannot be loaded."""


class DiarizationError(RuntimeError):
    """Raised when speaker diarization fails."""


@lru_cache(maxsize=2)
def _load_pipeline(token: str, device: str) -> Pipeline:
    """Load and cache the local Community-1 pipeline."""

    try:
        pipeline = Pipeline.from_pretrained(PIPELINE_ID, token=token)
        if pipeline is None:
            raise RuntimeError("Pipeline.from_pretrained returned None")
        pipeline.to(torch.device(device))
        return pipeline
    except Exception as exc:
        raise DiarizationModelError(
            "Не удалось загрузить pyannote Community-1. Проверьте HF_TOKEN и доступ "
            "к модели на Hugging Face."
        ) from exc


class PyannoteDiarizer:
    """Reusable speaker diarizer using the installed pyannote.audio API."""

    def __init__(self, device: str | None = None) -> None:
        token = os.getenv("HF_TOKEN")
        if not token:
            raise MissingHFTokenError(
                "Переменная окружения HF_TOKEN не задана или пуста."
            )

        self._token = token
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def pipeline(self) -> Pipeline:
        """Return the cached pipeline, loading it on first use."""

        return _load_pipeline(self._token, self.device)

    def diarize(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> list[DiarizationSegment]:
        """Return exclusive, non-overlapping speaker turns for decoded mono audio."""

        waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
        audio_in_memory = {"waveform": waveform, "sample_rate": sample_rate}

        try:
            output = self.pipeline(audio_in_memory)
            annotation = getattr(output, "exclusive_speaker_diarization", None)
            if annotation is None:
                annotation = getattr(output, "speaker_diarization", output)

            return [
                DiarizationSegment(
                    start=float(turn.start),
                    end=float(turn.end),
                    speaker=str(speaker),
                )
                for turn, _, speaker in annotation.itertracks(yield_label=True)
            ]
        except DiarizationModelError:
            raise
        except Exception as exc:
            raise DiarizationError("Не удалось выполнить диаризацию аудио.") from exc
