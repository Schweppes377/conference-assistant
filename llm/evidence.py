"""Conservative evidence rendering. No keyword-based semantic certification.

LLM fields are proposals. Quotes establish provenance only; applicability and
completeness of conditions still require review. Never emit model-generated
free prose as a verified compliance decision.
"""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from decimal import Decimal
from typing import Any

from llm.models import CitationValidationError, ContextValidationError, EvidenceAssessment, SourceEvidence
from llm.prompts import INSUFFICIENT_CONTEXT_ANSWER, PARTIAL_CONTEXT_MARKER

LIMITATION = (
    "Проверено происхождение выдержек и значений, но не их смысловая применимость "
    "к проекту и не полнота обязательных условий и исключений. Соответствие проекта "
    "нормативным требованиям автоматически не подтверждено."
)
_APPROXIMATE = re.compile(r"\b(?:около|примерно|приблизительно|порядка|предварительно|ориентировоч\w*)\b|[≈~]", re.I)


def _normalized(text: str) -> str:
    # Only whitespace: do not discard punctuation/negations or reorder clauses.
    return " ".join(text.split())


def _quote(value: Any, text: str, name: str, *, optional: bool = False) -> str:
    if optional and value is None:
        return ""
    if not isinstance(value, str) or (not value.strip() and not optional):
        raise CitationValidationError(f"{name}: требуется непустая дословная выдержка")
    if value.strip() and _normalized(value) not in _normalized(text):
        raise CitationValidationError(f"{name}: выдержка не найдена в указанном основании")
    return value.strip()


def _list(data: dict, key: str, item_type: type) -> list:
    value = data.get(key)
    if not isinstance(value, list) or len(value) > 30 or any(not isinstance(x, item_type) for x in value):
        raise CitationValidationError(f"{key}: ожидается список до 30 элементов")
    return value


def _fields(data: dict, required: set[str], optional: set[str] = frozenset()) -> None:
    if not required <= data.keys() or data.keys() - required - optional:
        raise CitationValidationError("Нарушена схема нормативного обоснования")


def _has_clause_heading(text: str, clause: str) -> bool:
    """Require a numbered heading/reference, not a decimal measurement."""
    marker = re.escape(clause)
    heading = r"(?:^|\n|[.!?]\s+)" + marker + r"(?:\.(?!\d)|\s+(?=[А-ЯЁA-Z]))"
    reference = r"\b(?:пункт(?:а|е|у)?|п\.)\s*" + marker + r"(?!\w|\.\d)"
    return bool(re.search(heading, text) or re.search(reference, text, re.I))


def compare_values(project: Decimal, required: Decimal, operator: str, *,
                   exact: bool, applicability_verified: bool,
                   conditions_verified: bool) -> str:
    """Only a trusted caller, NOT an LLM field, may attest applicability.

    The current automatic pipeline has no such semantic verifier and passes
    False. A matching number or string is intentionally not an attestation.
    """
    if not isinstance(operator, str) or operator not in {">=", "<=", "=", ">", "<"}:
        raise CitationValidationError("Неизвестный оператор сравнения")
    if not (project.is_finite() and required.is_finite()):
        raise CitationValidationError("Недопустимое числовое значение")
    if not (exact and applicability_verified and conditions_verified):
        return "not_established"
    valid = {">=": project >= required, "<=": project <= required,
             "=": project == required, ">": project > required, "<": project < required}[operator]
    return "satisfied" if valid else "violated"


def _single_measure(text: Any, evidence: str, name: str):
    from llm.service import _measurements
    if not isinstance(text, str):
        raise CitationValidationError(f"{name}: требуется значение с единицей")
    values = _measurements(text)
    if len(values) != 1 or not values <= _measurements(evidence):
        raise CitationValidationError(f"{name}: значение не подтверждено именно своей выдержкой")
    return next(iter(values))


def parse_evidence(content: str, question: str, sources: Sequence[SourceEvidence]) -> EvidenceAssessment | None:
    """None means legacy prose. Malformed JSON is never silently downgraded."""
    raw = content.strip()
    if not raw.startswith(("{", "[", "```")):
        return None
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)
    try:
        data = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise CitationValidationError("Невалидный JSON нормативного обоснования") from exc
    if not isinstance(data, dict):
        raise CitationValidationError("Нормативное обоснование должно быть JSON-объектом")
    _fields(data, {"project_facts", "requirements", "comparisons", "unknowns", "conclusion"})
    if not isinstance(data['conclusion'], str) or data['conclusion'] not in {"requirement_found", "requires_clarification", "insufficient_information", "potential_noncompliance", "compliant"}:
        raise CitationValidationError("Неизвестный conclusion")
    source_map = {s.source_id: s for s in sources}
    facts = []
    for item in _list(data, "project_facts", dict):
        _fields(item, {"quote", "certainty"})
        text = _quote(item['quote'], question, "project_fact")
        if not isinstance(item['certainty'], str) or item['certainty'] not in {"exact", "approximate", "unknown"}:
            raise CitationValidationError("Неизвестная точность проектного параметра")
        # Scan the entire question too: omitting 'около' from a quote must not
        # convert an approximate project value into an exact one.
        certainty = "approximate" if _APPROXIMATE.search(question) else item['certainty']
        facts.append({"quote": text, "certainty": certainty, "origin": "question"})
    requirements = []
    for item in _list(data, "requirements", dict):
        _fields(item, {"source_id", "quote", "scope_quote", "conditions_quote", "exceptions_quote", "clause"})
        sid = item['source_id']
        if not isinstance(sid, str) or sid not in source_map:
            raise CitationValidationError("Неизвестный SOURCE ID в нормативном обосновании")
        source = source_map[sid]
        text = _quote(item['quote'], source.text, "requirement")
        record = {"source_id": sid, "quote": text, "chunk_id": source.chunk_id,
                  "applicability": "unverified", "conditions_complete": "unverified"}
        for field in ("scope_quote", "conditions_quote", "exceptions_quote"):
            record[field] = _quote(item[field], source.text, field, optional=True)
        clause = item['clause']
        if clause is not None:
            if (not isinstance(clause, str) or not re.fullmatch(r"\d+(?:\.\d+)+", clause)
                    or not _has_clause_heading(text, clause)):
                raise CitationValidationError("Номер пункта не установлен в процитированной выдержке")
        record['clause'] = clause
        requirements.append(record)
    comparisons = []
    for item in _list(data, "comparisons", dict):
        _fields(item, {"project_fact", "requirement", "project_measure", "required_measure", "operator"})
        fi, ri = item['project_fact'], item['requirement']
        if type(fi) is not int or type(ri) is not int or not 0 <= fi < len(facts) or not 0 <= ri < len(requirements):
            raise CitationValidationError("Некорректные индексы оснований сравнения")
        project = _single_measure(item['project_measure'], facts[fi]['quote'], "project_measure")
        required = _single_measure(item['required_measure'], requirements[ri]['quote'], "required_measure")
        if project[0] != required[0]:
            raise ContextValidationError("Сравниваются разные размерности")
        outcome = compare_values(project[1], required[1], item['operator'],
                                 exact=facts[fi]['certainty'] == 'exact',
                                 applicability_verified=False, conditions_verified=False)
        comparisons.append({**item, "dimension": project[0], "project_canonical": str(project[1]),
                            "required_canonical": str(required[1]), "result": outcome,
                            "reason": "Применимость, оператор и полнота условий требуют независимой проверки"})
    unknowns = _list(data, "unknowns", str)
    if any(not text.strip() for text in unknowns):
        raise CitationValidationError("Пустое описание неизвестного параметра")
    if not requirements:
        conclusion = 'insufficient_information'
    elif data['conclusion'] != 'requirement_found' or comparisons or facts or unknowns:
        conclusion = 'requires_clarification'
    else:
        conclusion = 'requirement_found'
    return EvidenceAssessment(1, 'extractive', conclusion, facts, requirements,
                              comparisons, unknowns, [LIMITATION],
                              proposed_conclusion=data['conclusion'])


def legacy_assessment(sources: Sequence[SourceEvidence], *, insufficient: bool) -> EvidenceAssessment:
    """Keep old response/checkpoint format readable without certifying prose."""
    return EvidenceAssessment(1, 'legacy_extractive',
        'insufficient_information' if insufficient else 'requires_clarification', [],
        [{"source_id": s.source_id, "quote": s.text, "chunk_id": s.chunk_id,
          "clause": None, "applicability": "unverified", "conditions_complete": "unverified"}
         for s in sources], [], [], [LIMITATION, "Свободное толкование старого формата не проверено и не используется как итоговый нормативный вывод."])


def render_evidence(assessment: EvidenceAssessment, sources: Sequence[SourceEvidence]) -> str:
    """Only original source text and fixed local conclusions reach the answer.

    Full cited chunks preserve conditions potentially omitted in an LLM quote.
    Unknowns/model proposals remain in diagnostics, not as established facts.
    """
    if assessment.conclusion == 'insufficient_information':
        return INSUFFICIENT_CONTEXT_ANSWER
    lines = ["Найдены следующие нормативные фрагменты. Их применимость к проекту требует проверки."]
    if assessment.project_facts:
        lines += ["", "Данные из вопроса (не нормативные требования):"]
        for fact in assessment.project_facts:
            label = {"exact": "заявлено точным, независимая проверка не выполнена",
                     "approximate": "приблизительно", "unknown": "точность неизвестна"}[fact['certainty']]
            lines += [f"- {fact['quote']} ({label})."]
    for source in sources:
        lines += ["", f"[{source.source_id}] Дословный текст основания:", "",
                  *["> " + line for line in source.text.splitlines()]]
    lines += ["", LIMITATION]
    if assessment.unknowns:
        lines += ["", "Модель предлагает уточнить следующие сведения; полнота этого перечня не проверена:",
                  *["> " + line for item in assessment.unknowns for line in item.splitlines()]]
    if assessment.conclusion == 'requires_clarification':
        lines += ["", "Требуется уточнение проектных параметров, области применения, обязательных условий и исключений.",
                  PARTIAL_CONTEXT_MARKER]
    return "\n".join(lines)
