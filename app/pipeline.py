"""End-to-end audio transcription and speaker attribution."""

from __future__ import annotations

from pathlib import Path

from asr import WhisperTranscriber
from diarization import PyannoteDiarizer
from utils.audio import SAMPLE_RATE, load_audio

from .transcript import TranscriptSegment, merge_transcript


class MeetingTranscriptionPipeline:
    """Run decoding, ASR, diarization, and timestamp-based merging."""

    def __init__(
        self,
        model_name: str = "medium",
        language: str | None = "ru",
    ) -> None:
        self.transcriber = WhisperTranscriber(
            model_name=model_name,
            language=language,
        )
        self.diarizer = PyannoteDiarizer()

    def process(self, audio_path: str | Path) -> list[TranscriptSegment]:
        """Process one local audio file and return a speaker-attributed transcript."""

        audio = load_audio(audio_path, sample_rate=SAMPLE_RATE)
        asr_segments = self.transcriber.transcribe(audio)
        diarization_segments = self.diarizer.diarize(audio, SAMPLE_RATE)
        return merge_transcript(asr_segments, diarization_segments)
