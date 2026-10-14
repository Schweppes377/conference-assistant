"""Persistence and human-readable Markdown reporting for meeting analysis."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from app.analysis_models import (
    AnalysisOutputPaths,
    MeetingAnalysis,
    STATUS_INSUFFICIENT_INFORMATION,
    STATUS_PARTIAL_INFORMATION,
    STATUS_POTENTIAL_NONCOMPLIANCE,
    STATUS_PROCESSING_ERROR,
    STATUS_PENDING,
    STATUS_REQUIREMENT_FOUND,
)


_STATUS_LABELS = {
    STATUS_REQUIREMENT_FOUND: "Нормативное требование найдено",
    STATUS_PARTIAL_INFORMATION: "Нормативное требование найдено частично",
    STATUS_INSUFFICIENT_INFORMATION: "Недостаточно информации в нормативном корпусе",
    STATUS_POTENTIAL_NONCOMPLIANCE: "Обнаружено потенциальное несоответствие",
    STATUS_PROCESSING_ERROR: "Ошибка обработки вопроса",
    STATUS_PENDING: "Ожидает продолжения обработки",
}

_CATEGORY_LABELS = {
    "fire_safety": "Пожарная безопасность",
    "evacuation": "Эвакуация",
    "accessibility": "Доступность для МГН",
    "ventilation": "Вентиляция",
    "noise": "Защита от шума",
    "structural": "Конструкции",
    "utilities": "Инженерные системы",
    "other": "Другое",
}


def save_meeting_analysis(
    analysis: MeetingAnalysis,
    results_dir: str | Path,
) -> AnalysisOutputPaths:
    """Save transcript, complete JSON, and Markdown under unique filenames."""

    directory = Path(results_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    meeting_name = _safe_name(Path(analysis.audio_file).stem)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prefix = f"{meeting_name}_{run_id}"
    transcript_path = directory / f"{prefix}_transcript.json"
    analysis_path = directory / f"{prefix}_analysis.json"
    report_path = directory / f"{prefix}_report.md"

    transcript_payload = [asdict(segment) for segment in analysis.transcript]
    transcript_path.write_text(
        json.dumps(transcript_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    analysis_path.write_text(
        json.dumps(analysis.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(_build_report(analysis), encoding="utf-8")
    return AnalysisOutputPaths(
        transcript=str(transcript_path),
        analysis=str(analysis_path),
        report=str(report_path),
    )


def _build_report(analysis: MeetingAnalysis) -> str:
    stats = analysis.statistics
    lines = [
        "# Анализ совещания по нормативным вопросам",
        "",
        f"- Аудио: `{analysis.audio_file}`",
        f"- Создано: {analysis.created_at}",
        f"- Реплик в стенограмме: {stats.transcript_segments}",
        f"- Спикеры: {', '.join(stats.speakers) or 'не определены'}",
        f"- Выявлено нормативных вопросов: {stats.detected_issues}",
        f"- Найдено требований: {stats.requirement_found}",
        f"- Частично найдено требований: {stats.partial_information}",
        f"- Недостаточно информации: {stats.insufficient_information}",
        f"- Потенциальных несоответствий: {stats.potential_noncompliance}",
        f"- Ошибок обработки вопросов: {stats.processing_errors}",
        f"- Ожидают обработки: {stats.pending_issues}",
        "",
    ]
    if analysis.stop_reason:
        lines.extend([analysis.stop_reason, "", f"Файл продолжения: `{analysis.checkpoint_file}`", ""])
    if not analysis.detected_issues:
        lines.extend([("Detection ещё не завершён; стенограмма и выполненные окна сохранены."
                       if analysis.run_status != "completed" else "Нормативно значимые фрагменты не обнаружены."), ""])
        return "\n".join(lines)

    for issue in analysis.detected_issues:
        lines.extend(
            [
                "---",
                "",
                f"## {_timestamp(issue.start)} — {_timestamp(issue.end)} · "
                f"{_CATEGORY_LABELS.get(issue.category, issue.category)}",
                "",
                f"**Спикеры:** {', '.join(issue.speakers)}",
                "",
                "### Фрагмент совещания",
                "",
                *[f"> {line}" for line in issue.original_text.splitlines()],
                "",
                "### Выявленный вопрос",
                "",
                issue.normalized_question,
                "",
                f"**Причина проверки:** {issue.reason}",
                "",
                "### Ответ",
                "",
                issue.answer,
                "",
                f"**Статус:** {_STATUS_LABELS.get(issue.status, issue.status)}",
                "",
                "### Источники",
                "",
            ]
        )
        if issue.cited_sources:
            for source in issue.cited_sources:
                records = (issue.validation or {}).get("requirements", [])
                clauses = list(dict.fromkeys(r.get("clause") for r in records
                               if r.get("source_id") == source.source_id and r.get("clause")))
                lines.extend(
                    [
                        f"**[{source.source_id}] {source.document}**",
                        f"   - Пункт: {', '.join(clauses) if clauses else 'не установлен'}",
                        f"   - Обозначение: {source.designation or '—'}",
                        f"   - Страница PDF: {source.page}",
                        f"   - Идентификатор фрагмента: `{source.chunk_id}`",
                        f"   - Релевантность: {source.score:.4f}; "
                        f"расстояние: {source.distance:.4f}",
                        (
                            f"   - Оценка повторного ранжирования: "
                            f"{source.rerank_score:.4f}"
                            if source.rerank_score is not None
                            else "   - Оценка повторного ранжирования: —"
                        ),
                    ]
                )
                lines.extend(["", "Использованный нормативный фрагмент (полный текст chunk):", "",
                              *["> " + line for line in source.text.splitlines()], ""])
        else:
            lines.append("Использованные нормативные источники отсутствуют.")
        if issue.error:
            lines.extend(["", f"**Ошибка:** {issue.error}"])
        if issue.validation is None and issue.cited_sources:
            lines.extend(["", "Примечание: сохранённый ответ не имеет данных новой проверки обоснований. "
                          "Его статус не подтверждает соответствие проекта; исторический текст не переоценивался."])
        lines.append("")

    timings = stats.timings
    lines.extend(
        [
            "---",
            "",
            "## Время обработки",
            "",
            f"- Распознавание речи: {timings.asr_seconds:.3f} с",
            f"- Диаризация: {timings.diarization_seconds:.3f} с",
            f"- Выявление вопросов (LLM): {timings.detection_llm_seconds:.3f} с",
            f"- Поиск по нормативному корпусу: {timings.retrieval_seconds:.3f} с",
            f"- Формирование ответов (LLM): {timings.answer_llm_seconds:.3f} с",
            f"- Всего: {timings.total_seconds:.3f} с",
            "",
            "Статус «потенциальное несоответствие» не является юридическим "
            "заключением и требует проверки профильным специалистом.",
            "",
        ]
    )
    return "\n".join(lines)


def _timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE).strip("._")
    return cleaned or "meeting"
