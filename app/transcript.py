"""Timestamp-overlap matching for ASR and diarization results."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from asr import ASRSegment
from diarization import DiarizationSegment

UNKNOWN_SPEAKER = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """A recognized text segment assigned to a speaker."""

    start: float
    end: float
    speaker: str
    text: str


def _overlap_duration(
    first_start: float,
    first_end: float,
    second_start: float,
    second_end: float,
) -> float:
    return max(0.0, min(first_end, second_end) - max(first_start, second_start))


def merge_transcript(
    asr_segments: Iterable[ASRSegment],
    diarization_segments: Iterable[DiarizationSegment],
) -> list[TranscriptSegment]:
    """Assign each ASR segment to the speaker with the largest total overlap."""

    speaker_turns = list(diarization_segments)
    result: list[TranscriptSegment] = []

    for speech in asr_segments:
        overlap_by_speaker: dict[str, float] = defaultdict(float)
        for turn in speaker_turns:
            overlap = _overlap_duration(
                speech.start,
                speech.end,
                turn.start,
                turn.end,
            )
            if overlap > 0:
                overlap_by_speaker[turn.speaker] += overlap

        speaker = UNKNOWN_SPEAKER
        if overlap_by_speaker:
            speaker = max(
                sorted(overlap_by_speaker),
                key=overlap_by_speaker.__getitem__,
            )

        result.append(
            TranscriptSegment(
                start=speech.start,
                end=speech.end,
                speaker=speaker,
                text=speech.text,
            )
        )

    return result
