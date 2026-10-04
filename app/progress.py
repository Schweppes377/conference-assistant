"""Small JSON checkpoints and import of existing transcript/analysis files."""

from __future__ import annotations

import json
import math
import os
from dataclasses import fields
from pathlib import Path
from typing import Any

from app.analysis_models import DetectedIssue, STATUS_PENDING, STATUS_PROCESSING_ERROR


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    """Atomically replace the checkpoint without exposing environment secrets."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_input(path: Path, *, mode: str) -> dict[str, Any]:
    """Import JSON; analysis mode preserves completed answers and candidates."""
    with path.open(encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if mode == "resume":
        if not isinstance(data, dict) or data.get("checkpoint_version") != 1:
            raise ValueError("Для --resume нужен *_progress.json версии 1.")
        _validate_transcript(data.get("transcript"))
        return data
    raw_transcript = data if isinstance(data, list) else data.get("transcript")
    transcript = _validate_transcript(raw_transcript)
    state: dict[str, Any] = {
        "checkpoint_version": 1, "audio_file": (
            data.get("audio_file", str(path)) if isinstance(data, dict) else str(path)
        ), "transcript": transcript, "answers": {}, "prepared": {},
        "detection_progress": {}, "detection_complete": False,
    }
    if mode == "issues":
        if not isinstance(data, dict) or not isinstance(data.get("detected_issues"), list):
            raise ValueError("Для --issues нужен analysis.json с detected_issues и transcript.")
        names = {field.name for field in fields(DetectedIssue)}
        issues = []
        for index, raw in enumerate(data["detected_issues"], 1):
            values = {name: raw[name] for name in names if name in raw}
            values.setdefault("search_terms", [])
            values.setdefault("id", f"ISSUE_{index:03d}")
            try:
                issue = DetectedIssue(**values)
            except TypeError as exc:
                raise ValueError("Некорректный detected issue в analysis.json") from exc
            if not issue.normalized_question.strip():
                raise ValueError("Вопрос не может быть пустым")
            issues.append(values)
            if raw.get("status") not in (None, STATUS_PENDING, STATUS_PROCESSING_ERROR):
                state["answers"][issue.id] = raw
            prepared = data.get("prepared", {}).get(issue.id)
            if prepared:
                state["prepared"][issue.id] = prepared
        if len({i["id"] for i in issues}) != len(issues):
            raise ValueError("Идентификаторы detected issues должны быть уникальными")
        state["detected_issues"] = issues
        state["detection_complete"] = True
        # Legacy totals may omit rejected responses; retain them separately.
        state["historical_usage"] = data.get("statistics", {})
        state["llm_provider"] = data.get("llm_provider", "unknown")
        state["llm_model"] = data.get("llm_model", "unknown")
    return state


def check_llm_identity(state: dict[str, Any], *, provider: str, model: str) -> None:
    """Reject mixed paid results, including legacy checkpoints without identity."""
    identity = (state.get("llm_provider"), state.get("llm_model"))
    if identity == (provider, model):
        return
    runtime = state.get("llm_runtime", {})
    has_work = bool(state.get("answers") or runtime.get("responses")
                    or state.get("detection_progress")
                    or any(e.get("successful") for e in runtime.get("events", [])))
    if has_work:
        raise ValueError(
            "Сохранённые LLM-результаты созданы другой или неизвестной моделью/provider "
            f"({identity[0] or 'unknown'} / {identity[1] or 'unknown'}). "
            "Нельзя смешивать эксперименты. Начните заново через --transcript <progress.json> "
            "или --issues <analysis.json> --reanswer без повторного ASR/diarization."
        )
    # No successful generations to preserve: safely adopt current provider.
    state["llm_provider"], state["llm_model"] = provider, model
    if state.get("settings"):
        state["settings"]["qa"]["model"] = model
    runtime.get("events", []).clear()


def _validate_transcript(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("Нужна непустая JSON-стенограмма (start/end/speaker/text).")
    result = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Некорректный сегмент стенограммы")
        try:
            start, end = float(item["start"]), float(item["end"])
            text = item["text"]
            speaker = item.get("speaker", "UNKNOWN")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Некорректный сегмент стенограммы") from exc
        if (not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start
                or not isinstance(text, str) or not isinstance(speaker, str)):
            raise ValueError("Некорректные timestamps/text/speaker стенограммы")
        result.append({"start": start, "end": end, "speaker": speaker, "text": text})
    return result
