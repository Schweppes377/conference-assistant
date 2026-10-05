"""Synthetic fixtures, independent of construction baseline output."""
import copy
import json
from pathlib import Path
import pytest
import jsonschema
from evaluation.content_metrics import speaker_attribution_metrics, text_similarity
from evaluation.metrics import detection_metrics
from evaluation.data import validate_dataset
from evaluation.run import evaluate, save_results

def turn(text,speaker,id): return dict(text=text,speaker=speaker,id=id)
def hypothesis(text,speaker): return dict(text=text,speaker=speaker,start=100,end=101)

def test_permuted_labels_and_split_segments():
    refs=[turn('кот ест рыбу','REF_SPEAKER_01','a'),turn('собака гуляет дома','REF_SPEAKER_02','b')]
    hyps=[hypothesis('кот','Y'),hypothesis('ест рыбу','Y'),hypothesis('собака гуляет дома','X')]
    r=speaker_attribution_metrics(refs,hyps)
    assert r['speaker_attribution_accuracy']==1
    assert r['speaker_mapping']=={'Y':'REF_SPEAKER_01','X':'REF_SPEAKER_02'}
    assert r['matched_utterances']==2

def test_wrong_attribution_cannot_be_repaired_by_per_turn_mapping():
    refs=[turn('первый текст','REF_SPEAKER_01','a'),turn('второй ответ','REF_SPEAKER_02','b'),turn('третий вопрос','REF_SPEAKER_01','c')]
    hyps=[hypothesis(t['text'],'X') for t in refs]
    r=speaker_attribution_metrics(refs,hyps)
    assert r['correct_utterances']==2 and r['incorrect_utterances']==1
    assert r['speaker_attribution_accuracy']==pytest.approx(2/3)

def test_missing_text_is_unmatched_not_in_accuracy_denominator():
    refs=[turn('кот ест рыбу','REF_SPEAKER_01','a'),turn('собака гуляет','REF_SPEAKER_02','b')]
    r=speaker_attribution_metrics(refs,[hypothesis('кот ест рыбу','X')])
    assert r['matched_utterances']==1 and r['unmatched_utterances']==1
    assert r['matching_coverage']==.5 and r['speaker_attribution_accuracy']==1

def test_unknown_speaker_is_not_mapped_to_known_role():
    r=speaker_attribution_metrics([turn('кот ест','REF_SPEAKER_01','a')],[hypothesis('кот ест','UNKNOWN')])
    assert r['incorrect_utterances']==1 and r['speaker_attribution_accuracy']==0

def test_tied_word_votes_are_incorrect_not_optimistically_assigned():
    r=speaker_attribution_metrics([turn('кот ест','REF_SPEAKER_01','a')],[hypothesis('кот','X'),hypothesis('ест','Y')])
    assert r['matched_utterances']==1 and r['incorrect_utterances']==1

def test_word_alignment_is_deterministic_and_ignores_time_values():
    refs=[turn('да да вопрос','REF_SPEAKER_01','a')]
    h=[hypothesis('да вопрос','X')]
    a=speaker_attribution_metrics(refs,h)
    h[0].update(start=-999,end=999999)
    assert a==speaker_attribution_metrics(refs,h)

def test_content_matching_no_times_duplicate_false_positive():
    refs=[dict(id='r',question='Минимальная ширина лестничного марша?',match_concepts=[['лестнич','лестниц'],['ширин']])]
    preds=[dict(normalized_question='Какая ширина лестницы допускается?')]*2
    r=detection_metrics(refs,preds,mode='text',threshold=.2)
    assert (r['tp'],r['fp'],r['fn'])==(1,1,0)
    assert r['precision']==.5
    assert r==detection_metrics(refs,[dict(p,start=-100,end=999) for p in preds],mode='text',threshold=.2)

def test_concept_gate_rejects_other_evacuation_topic():
    refs=[dict(id='r',question='Какая высота эвакуационного выхода?',match_concepts=[['высот'],['выход','двер']])]
    r=detection_metrics(refs,[dict(normalized_question='Какая ширина эвакуационного выхода?')],mode='text',threshold=.2)
    assert (r['tp'],r['fp'],r['fn'])==(0,1,1)

def test_similar_word_inflections_and_empty_strings():
    assert text_similarity('вентиляция жилого здания','вентиляции жилых зданий')>.5
    assert text_similarity('','текст')==0

def dataset():
    return {'schema_version':1,'id':'synthetic','audio_file':'unused.wav',
        'reference_transcript':'ширина лестницы',
        'reference_utterances':[dict(turn('ширина лестницы','REF_SPEAKER_01','t'),confirmed=True)],
        'reference_issues':[dict(id='r',question='ширина лестницы',confirmed=True,expected_sources=[dict(document='DOC',page=7,chunk_id='c')])],
        'matching':dict(mode='text',threshold=.2),'verification':dict(status='verified')}

def analysis():
    return {'transcript':[hypothesis('ширина лестницы','X')],'statistics':{'timings':{},'llm_usage':{}},'audio_duration_sec':10,
        'detected_issues':[dict(id='p',normalized_question='ширина лестницы',start=100,end=101,answer='',status='ok',cited_sources=[],
            retrieved_candidates=[dict(source_id='s',document='DOC',page=7,chunk_id='c',text='норма')])]}

def test_content_integration_retrieval_and_export_without_der(tmp_path,monkeypatch):
    import evaluation.run
    monkeypatch.setattr(evaluation.run,'diarization_metrics',lambda *a,**k:pytest.fail('DER must not be called'))
    r=evaluate(dataset(),analysis())
    assert r['asr']['wer']==0 and r['diarization']['speaker_attribution_accuracy']==1
    assert r['question_detection']['f1']==1 and r['retrieval']['hit_at_1']==1
    assert 'der' not in r['diarization']
    save_results(r,tmp_path)
    assert (tmp_path/'speaker_attribution.csv').is_file()

def test_pending_verification_blocks_all_quality_scores():
    d=dataset();d['verification']['status']='pending'
    r=evaluate(d,analysis())
    assert r['asr']['wer'] is None and r['diarization']['speaker_attribution_accuracy'] is None
    assert r['question_detection']['tp'] is None and r['retrieval']['hit_at_1'] is None

def test_verified_text_must_equal_confirmed_turns():
    d=dataset();d['reference_transcript']='другой текст'
    with pytest.raises(ValueError):validate_dataset(d)
    d=dataset();d['reference_utterances'][0]['confirmed']=False
    with pytest.raises(ValueError):validate_dataset(d)

def test_schema_new_and_legacy_formats():
    schema=json.loads((Path(__file__).parent/'datasets/dataset.schema.json').read_text(encoding='utf-8'))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(dataset(),schema);validate_dataset(dataset())
    d=dict(schema_version=1,id='old',audio_file='old.wav',reference_diarization=[dict(start=0,end=1,speaker='A')])
    jsonschema.validate(d,schema);validate_dataset(d)

def test_expected_sources_remain_frozen_in_new_dataset():
    folder=Path(__file__).parent/'datasets'
    d=json.loads((folder/'construction_meeting.json').read_text(encoding='utf-8'))
    p=json.loads((folder/'construction_meeting.reference_preparation.json').read_text(encoding='utf-8'))
    assert len(d['reference_utterances'])==27 and len(d['reference_issues'])==8
    assert [i['expected_sources'] for i in d['reference_issues']]==[i['expected_sources'] for i in p['reference_issues']]
    assert all('start' not in i and 'end' not in i for i in d['reference_issues'])
    jsonschema.validate(d,json.loads((folder/'dataset.schema.json').read_text(encoding='utf-8')))
    validate_dataset(d)
