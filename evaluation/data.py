"""Strict evaluation inputs and measured audio metadata (no decoding/models)."""

from __future__ import annotations

import json
import math
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Некорректный JSON: {path.name}, строка {error.lineno}") from None


def resolve_file(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def validate_turns(turns: list, *, text_required: bool = False) -> None:
    if not isinstance(turns, list):
        raise ValueError("Annotation/transcript must be an array")
    for turn in turns:
        if not isinstance(turn, dict):
            raise ValueError("Annotation turn must be an object")
        if text_required and not isinstance(turn.get("text"), str):
            raise ValueError("Transcript text must be a string")
        for name in ("start", "end"):
            value = turn.get(name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError("Annotation timestamps must be finite numbers")
        if turn["start"] < 0 or turn["end"] < turn["start"]:
            raise ValueError("Invalid annotation interval")
        if not isinstance(turn.get("speaker"), str) or not turn["speaker"]:
            raise ValueError("Speaker label is required")


def validate_dataset(data: dict) -> None:
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Evaluation dataset must be an object with schema_version=1")
    matching = data.get("matching", {})
    if (not isinstance(matching, dict) or matching.get("mode", "time") not in ("time", "segments")
            or not isinstance(matching.get("threshold", .5), (int, float))
            or not 0 < matching.get("threshold", .5) <= 1):
        raise ValueError("Invalid issue matching policy")
    der = data.get("der", {})
    if (not isinstance(der, dict) or not isinstance(der.get("collar_sec", .25), (int, float))
            or not math.isfinite(der.get("collar_sec", .25)) or der.get("collar_sec", .25) < 0):
        raise ValueError("Invalid DER collar")
    for name in ("reference_transcript", "reference_diarization"):
        value = data.get(name)
        if value is not None:
            if name == "reference_transcript" and isinstance(value, str):
                continue
            validate_turns(value, text_required=name == "reference_transcript")
    issues = data.get("reference_issues")
    if issues is not None:
        if not isinstance(issues, list) or any(not isinstance(i, dict) or not isinstance(i.get("id"), str) for i in issues):
            raise ValueError("reference_issues must contain objects with stable IDs")
        if len({i["id"] for i in issues}) != len(issues):
            raise ValueError("Duplicate reference issue IDs")
        mode = data.get("matching", {}).get("mode", "time")
        for issue in issues:
            if mode == "time":
                validate_turns([{**issue, "speaker": "reference"}])
                if issue["end"] <= issue["start"]:
                    raise ValueError("Reference issue interval must have positive duration")
            elif not isinstance(issue.get("segment_ids"), list) or not issue["segment_ids"]:
                raise ValueError("Segment matching requires nonempty segment_ids")
            if issue.get("expected_sources") is not None:
                if not isinstance(issue["expected_sources"], list):
                    raise ValueError("expected_sources must be an array")
                for source in issue["expected_sources"]:
                    if not isinstance(source, dict) or not (source.get("designation") or source.get("document")):
                        raise ValueError("Expected source needs document/designation")
    regions = data.get("evaluation_regions")
    if regions is not None:
        if not regions:
            raise ValueError("Explicit evaluation_regions cannot be empty")
        validate_turns([{**r, "speaker": "region"} for r in regions])


def validate_analysis(data: dict) -> None:
    if not isinstance(data, dict) or not isinstance(data.get("statistics"), dict):
        raise ValueError("Analysis must contain statistics, transcript and detected_issues")
    validate_turns(data.get("transcript"), text_required=True)
    issues = data.get("detected_issues")
    if not isinstance(issues, list):
        raise ValueError("Analysis detected_issues must be an array")
    for issue in issues:
        if not isinstance(issue, dict) or not all(isinstance(issue.get(k), str) for k in ("id", "normalized_question", "answer", "status")):
            raise ValueError("Incomplete analyzed issue")
        validate_turns([{**issue, "speaker": "issue"}])
        for name in ("cited_sources", "retrieved_candidates"):
            if not isinstance(issue.get(name, []), list):
                raise ValueError(f"{name} must be an array")
            for source in issue.get(name, []):
                if not isinstance(source, dict) or not all(source.get(k) for k in ("source_id", "document", "chunk_id", "text")) or not isinstance(source.get("page"), int):
                    raise ValueError("Incomplete source metadata in analysis")
    if len({i["id"] for i in issues}) != len(issues):
        raise ValueError("Duplicate analyzed issue IDs")


def audio_duration(path: Path | None) -> tuple[float | None, str | None]:
    if path is None or not path.is_file():
        return None, "audio file unavailable"
    try:
        with wave.open(str(path), "rb") as audio:
            return audio.getnframes() / audio.getframerate(), "WAV header"
    except (wave.Error, EOFError):
        return None, "non-PCM WAV/other format: provide measured audio_duration_sec"


def performance_metrics(analysis: dict, duration: float | None, duration_source: str | None) -> dict:
    timings = analysis["statistics"].get("timings", {})
    names = {"asr_sec": "asr_seconds", "diarization_sec": "diarization_seconds",
             "detection_sec": "detection_llm_seconds", "retrieval_sec": "retrieval_seconds",
             "answer_llm_sec": "answer_llm_seconds", "total_sec": "total_seconds"}
    result = {name: timings.get(key) for name, key in names.items()}
    for value in result.values():
        if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
            raise ValueError("Invalid measured latency")
    # Production uses zero placeholders for unexecuted stages. They are not
    # measurements of a zero-second inference and must not enter timing tables.
    result = {key: None if value == 0 else value for key, value in result.items()}
    usage = analysis["statistics"].get("llm_usage", {})
    resumed = usage.get("llm_calls") != usage.get("run_llm_calls") or bool(usage.get("historical_usage"))
    full_audio = (result["asr_sec"] or 0) > 0 and (result["diarization_sec"] or 0) > 0
    result.update(audio_duration_sec=duration, duration_source=duration_source,
                  asr_real_time_factor=result["asr_sec"] / duration if duration and full_audio else None,
                  full_real_time_factor=result["total_sec"] / duration if duration and full_audio and not resumed and result["total_sec"] is not None else None,
                  scope="saved/resumed stages; total is not cumulative wall time" if resumed else "single saved run",
                  reason=None if duration else "audio duration not measured")
    return result
