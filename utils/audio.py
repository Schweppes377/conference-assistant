"""Audio loading shared by ASR and diarization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from faster_whisper.audio import decode_audio

SAMPLE_RATE = 16_000


class AudioDecodingError(RuntimeError):
    """Raised when an input audio file cannot be decoded."""


def load_audio(path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode a local audio file to mono float32 PCM."""

    audio_path = Path(path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Аудиофайл не найден: {audio_path}")

    try:
        audio = np.asarray(
            decode_audio(str(audio_path), sampling_rate=sample_rate),
            dtype=np.float32,
        )
    except Exception as exc:
        raise AudioDecodingError(
            f"Файл не поддерживается или не читается: {audio_path}"
        ) from exc

    if audio.size == 0:
        raise AudioDecodingError(f"Аудиофайл пуст: {audio_path}")
    return audio
