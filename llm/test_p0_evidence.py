"""Synthetic P0 regression tests. No real documents, network or inference."""
import json
from dataclasses import asdict, replace
from decimal import Decimal
from pathlib import Path

import pytest

from llm.config import QAConfig
from llm.evidence import compare_values, parse_evidence, render_evidence
from llm.models import CitationValidationError, ContextValidationError, PreparedQuestion, SourceEvidence
from llm.service import QuestionAnsweringService, _resolve_citations, _validate_completion
from llm.test_service import FakeLLMClient, FakeRetriever


def source(text, sid="SOURCE_1"):
    return SourceEvidence(sid, "Синтетический норматив", "СП TEST", 2, sid + ':p2:c0', text, .95, .05)


def payload(text, *, quote=None, question=None, conclusion="requirement_found"):
    return {"project_facts": [] if question is None else [{"quote": question, "certainty": "exact"}],
            "requirements": [{"source_id": "SOURCE_1", "quote": quote or text,
                              "scope_quote": None, "conditions_quote": None,
                              "exceptions_quote": None, "clause": None}],
            "comparisons": [], "unknowns": [], "conclusion": conclusion}


def run(data, question, sources, *, first=None):
    content = json.dumps(data, ensure_ascii=False)
    client = FakeLLMClient([first, content] if first is not None else content)
    prepared = PreparedQuestion(question, question, sources, sources, 0.0)
    qa = QuestionAnsweringService(retriever=FakeRetriever([]), llm_client=client, config=QAConfig())
    result = qa.answer(question, prepared=prepared)
    return result, client


def test_a_exact_width_cannot_confirm_compliance():
    q = 'Проектная ширина 1,00 м.'
    text = 'Минимальная ширина прохода — 1,05 м.'
    data = payload(text, question=q, conclusion='compliant')
    data['comparisons'] = [{"project_fact": 0, "requirement": 0, "project_measure": "100 см",
                            "required_measure": "1050 мм", "operator": ">="}]
    result, client = run(data, q, [source(text)])
    assert result.validation['compliance_confirmed'] is False
    assert result.validation['proposed_conclusion'] == 'compliant'
    assert result.validation['comparisons'][0]['result'] == 'not_established'
    assert result.partial_context and client.calls == 1
    assert 'соответствие подтверждено' not in result.answer.casefold()
    # Arithmetic only after independent applicability and conditions attestation.
    assert compare_values(Decimal('1'), Decimal('1.05'), '>=', exact=True,
                          applicability_verified=True, conditions_verified=True) == 'violated'


@pytest.mark.parametrize('q', ['Ширина около метра.', 'Ширина примерно 1 м.', 'Ширина около 105 см.'])
def test_b_approximate_fact_never_becomes_exact(q):
    text = 'Ширина не менее 1,05 м.'
    data = payload(text, question=q, conclusion='compliant')
    result, _ = run(data, q, [source(text)])
    assert result.validation['project_facts'][0]['certainty'] == 'approximate'
    assert result.validation['conclusion'] == 'requires_clarification'
    assert not result.validation['compliance_confirmed']


def test_b_cannot_omit_approximation_word_from_project_quote():
    text = 'Ширина не менее 1,05 м.'
    data = payload(text, question='105 см', conclusion='compliant')
    result, _ = run(data, 'Ширина около 105 см.', [source(text)])
    assert result.validation['project_facts'][0]['certainty'] == 'approximate'


def test_c_unknown_mandatory_measures_prevent_permission():
    text = 'Допускается при площади до 600 м² и при наличии сигнализации.'
    q = 'Площадь 550 м². Сигнализация неизвестна.'
    data = payload(text, question=q, conclusion='compliant')
    data['requirements'][0]['conditions_quote'] = 'при наличии сигнализации'
    data['unknowns'] = ['Наличие сигнализации']
    result, _ = run(data, q, [source(text)])
    assert result.validation['conclusion'] == 'requires_clarification'
    assert text in result.answer
    assert not result.validation['compliance_confirmed']
    assert compare_values(Decimal(550), Decimal(600), '<=', exact=True,
                          applicability_verified=True, conditions_verified=False) == 'not_established'


@pytest.mark.parametrize('text', ['Ответ [SOURCE_1, SOURCE_999].', 'Ответ [SOURCE_999].',
                                 'Ответ [SOURCE_1; SOURCE_999].'])
def test_d_unknown_grouped_and_single_ids_rejected(text):
    with pytest.raises(CitationValidationError):
        _validate_completion(text, 'Вопрос', [source('Требование')])


def test_d_group_order_and_duplicates():
    sources = [source('Первый'), source('Второй', 'SOURCE_3')]
    assert [s.source_id for s in _resolve_citations('[ SOURCE_3, SOURCE_1 ] [SOURCE_3]', sources)] == ['SOURCE_3', 'SOURCE_1']


def test_e_number_must_belong_to_its_claim_sources():
    sources = [source('Минимальная ширина 1,05 м.'), source('Площадь определяется расчётом.', 'SOURCE_4')]
    with pytest.raises(CitationValidationError):
        _validate_completion('Минимальная ширина 1,05 м [SOURCE_4]. Другие сведения [SOURCE_1].', 'Ширина?', sources)


def test_e_question_value_is_not_a_normative_minimum():
    with pytest.raises(ContextValidationError):
        _validate_completion('Минимум 0,8 м [SOURCE_1].', 'Проектная ширина 0,8 м.', [source('Ширина определяется расчётом.')])


def test_e_structured_number_cannot_borrow_from_other_requirement():
    text = 'Минимальная ширина 1,05 м.'
    data = payload('Требование без числа', question='Ширина 1 м.')
    data['requirements'][0]['source_id'] = 'SOURCE_4'
    data['comparisons'] = [{'project_fact': 0, 'requirement': 0, 'project_measure': '1 м',
                            'required_measure': '1,05 м', 'operator': '>='}]
    with pytest.raises(CitationValidationError):
        parse_evidence(json.dumps(data, ensure_ascii=False), 'Ширина 1 м.', [source(text), source('Требование без числа', 'SOURCE_4')])


def test_f_and_cannot_be_rewritten_as_or():
    text = 'Объединение допускается при наличии клапана и отдельного канала.'
    data = payload(text, quote=text.replace(' и ', ' или '), conclusion='compliant')
    with pytest.raises(CitationValidationError):
        parse_evidence(json.dumps(data, ensure_ascii=False), 'Можно объединять?', [source(text)])


def test_f_short_quote_cannot_hide_remaining_conditions_in_output():
    text = 'Объединение допускается при наличии клапана и отдельного канала.'
    result, _ = run(payload(text, quote='Объединение допускается', conclusion='compliant'),
                    'Можно объединять?', [source(text)])
    assert text in result.answer
    assert result.validation['requirements'][0]['conditions_complete'] == 'unverified'
    assert not result.validation['compliance_confirmed']


def test_g_lighting_positive_information_survives():
    text = '4.3.12. На путях эвакуации следует предусматривать аварийное освещение.'
    data = payload(text)
    data['requirements'][0]['clause'] = '4.3.12'
    result, client = run(data, 'Нужно ли аварийное освещение на путях эвакуации?', [source(text)])
    assert not result.insufficient_context and not result.partial_context
    assert text in result.answer and '[SOURCE_1]' in result.answer
    assert result.validation['conclusion'] == 'requirement_found'
    assert result.validation['requirements'][0]['clause'] == '4.3.12'
    assert client.calls == 1


def test_decimal_measurement_is_not_a_clause_number():
    text = 'Минимальная ширина 1.05 м.'
    data = payload(text)
    data['requirements'][0]['clause'] = '1.05'
    with pytest.raises(CitationValidationError):
        parse_evidence(json.dumps(data, ensure_ascii=False), 'Ширина?', [source(text)])


@pytest.mark.parametrize('clause', ['4.3.12', '4.3.120'])
def test_clause_must_be_an_actual_heading_or_explicit_reference(clause):
    text = 'Высота 4.3.12 мм. Требование определяется расчётом.'
    data = payload(text)
    data['requirements'][0]['clause'] = clause
    with pytest.raises(CitationValidationError):
        parse_evidence(json.dumps(data, ensure_ascii=False), 'Размер?', [source(text)])


def test_legacy_false_permission_not_published_as_final_claim():
    q = 'Ширина около одного метра.'
    s = source('Минимальная ширина 1,05 м.')
    client = FakeLLMClient('Проект соответствует требованиям [SOURCE_1].')
    qa = QuestionAnsweringService(retriever=FakeRetriever([]), llm_client=client, config=QAConfig())
    result = qa.answer(q, prepared=PreparedQuestion(q, q, [s], [s], 0))
    assert 'Проект соответствует требованиям' not in result.answer
    assert result.validation['mode'] == 'legacy_extractive'
    assert result.partial_context and client.calls == 1


def test_search_terms_not_used_as_project_facts():
    q = 'Каков минимум?'
    s = source('Минимум определяется расчётом.')
    client = FakeLLMClient('Минимум 0,9 м [SOURCE_1].')
    qa = QuestionAnsweringService(retriever=FakeRetriever([]), llm_client=client, config=QAConfig())
    with pytest.raises(ContextValidationError):
        qa.answer(q, search_terms=['0,9 м'], prepared=PreparedQuestion(q, q + ' 0,9 м', [s], [s], 0))


def test_arbitrary_prose_citation_does_not_establish_full_support():
    cited, insufficient, partial = _validate_completion(
        'Любое отступление допускается без ограничений [SOURCE_1].', 'Можно отступить?',
        [source('Условия определяются проектом.')])
    assert cited and not insufficient and partial


def test_model_cannot_certify_its_own_applicability():
    text = 'Ширина не менее 1,05 м.'
    data = payload(text, conclusion='compliant')
    data['requirements'][0]['applicability_verified'] = True
    with pytest.raises(CitationValidationError):
        parse_evidence(json.dumps(data, ensure_ascii=False), 'Соответствует ли проект?', [source(text)])


def test_malformed_json_uses_one_corrective_generation_and_accounts_tokens():
    text = 'Нужно аварийное освещение.'
    result, client = run(payload(text), 'Освещение?', [source(text)], first='{broken')
    assert client.calls == 2 and result.total_tokens == 240


def test_two_invalid_json_responses_do_not_trigger_third_call():
    s = source('Требование')
    client = FakeLLMClient(['{bad', '{bad', 'unused'])
    qa = QuestionAnsweringService(retriever=FakeRetriever([]), llm_client=client, config=QAConfig())
    with pytest.raises(CitationValidationError):
        qa.answer('Вопрос', prepared=PreparedQuestion('Вопрос', 'Вопрос', [s], [s], 0))
    assert client.calls == 2


def test_units_and_dimensions_in_structured_comparison():
    text = 'Ширина не менее 105 см.'
    data = payload(text, question='Ширина 1050 мм.')
    data['comparisons'] = [{'project_fact': 0, 'requirement': 0, 'project_measure': '1.05 м',
                            'required_measure': '1,05 м', 'operator': '>='}]
    result, _ = run(data, 'Ширина 1050 мм.', [source(text)])
    assert result.validation['comparisons'][0]['project_canonical'] == '1.05'
    bad = payload('Площадь 1,05 м².', question='Ширина 1,05 м.')
    bad['comparisons'] = [{**data['comparisons'][0], 'required_measure': '1,05 м²'}]
    with pytest.raises(ContextValidationError):
        parse_evidence(json.dumps(bad, ensure_ascii=False), 'Ширина 1,05 м.', [source('Площадь 1,05 м².')])


def test_report_and_resume_keep_ids_text_and_validation(tmp_path):
    from app.test_api_budget import fixture, pipeline, FakeClient, FakeRetriever as MeetingRetriever
    from app.reporting import save_meeting_analysis
    from app.progress import read_input
    transcript, detected = fixture(1)
    text = '4.1. На пути требуется освещение.'
    data = payload(text)
    data['requirements'][0]['clause'] = '4.1'
    runner = pipeline(FakeClient([detected, json.dumps(data, ensure_ascii=False)]), MeetingRetriever(text),
                      checkpoint_path=tmp_path / 'progress.json')
    result = runner.analyze_transcript(transcript)
    issue = result.detected_issues[0]
    assert issue.validation['version'] == 1
    assert issue.status == 'requirement_found'
    # Deliberately non-positional IDs and reverse citation order, as in audit.
    first = replace(issue.cited_sources[0], source_id='SOURCE_2')
    second = replace(first, source_id='SOURCE_1')
    issue = replace(issue, cited_sources=[first, second])
    output = save_meeting_analysis(replace(result, detected_issues=[issue]), tmp_path)
    report = Path(output.report).read_text(encoding='utf-8')
    assert '**[SOURCE_2]' in report and '**[SOURCE_1]' in report
    assert '1. **Синтетический' not in report
    assert text in report and 'Пункт: 4.1' in report and 'Страница PDF:' in report
    state = read_input(tmp_path / 'progress.json', mode='resume')
    client = FakeClient([])
    resumed = pipeline(client, MeetingRetriever(text), state=state).analyze_transcript(transcript)
    assert client.prompts == []
    assert resumed.detected_issues[0].validation == result.detected_issues[0].validation


def test_old_analysis_without_validation_loads_without_reprocessing(tmp_path):
    from app.test_api_budget import fixture, pipeline, FakeClient, FakeRetriever as MeetingRetriever
    from app.progress import read_input
    transcript, detected = fixture(1)
    result = pipeline(FakeClient([detected, 'Требование [SOURCE_1].']), MeetingRetriever()).analyze_transcript(transcript)
    data = result.to_dict()
    for issue in data['detected_issues']:
        issue.pop('validation')
    path = tmp_path / 'old.json'
    path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    client = FakeClient([])
    resumed = pipeline(client, MeetingRetriever(), state=read_input(path, mode='issues')).analyze_transcript(transcript)
    assert client.prompts == []
    assert resumed.detected_issues[0].validation is None
