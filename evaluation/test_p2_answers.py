"""P2 recorder/runner tests use only synthetic completions, never API."""
import pytest
from llm.base import LLMClient, LLMCompletion, RateLimitError
from llm.config import QAConfig
from llm.runtime import RunLLMClient
from llm.service import QuestionAnsweringService
from evaluation.p2_answers import RecordingClient,run_questions,render_report,main
from rag.test_p1_retrieval import Retriever,hit


class Fake(LLMClient):
    model='GigaChat-3-Ultra'
    provider='gigachat'
    def __init__(self,fail=False): self.calls=0;self.fail=fail
    def complete(self,**kwargs):
        self.calls+=1
        if self.fail: raise RateLimitError('Synthetic rate limit',retry_after_seconds=120)
        return LLMCompletion('{',self.model,10,3,13)


def setup(fail=False,issues=1):
    state={'inputs':[{'id':f'TEST_{i}','normalized_question':'Высота ограждения галереи?',
                     'search_terms':['высота ограждения галереи']} for i in range(issues)],
           'results':{},'raw_attempts':[],'qa_config':{'model':'GigaChat-3-Ultra'}}
    fake=Fake(fail)
    recorder=RecordingClient(fake,state,lambda:None)
    runtime=RunLLMClient(recorder,max_calls=4)
    service=QuestionAnsweringService(retriever=Retriever([hit('4.1 Высота ограждения галереи должна быть 1,4 м.')]),
                                    llm_client=runtime,config=QAConfig())
    return state,fake,recorder,runtime,service


def test_rejected_json_is_saved_with_both_paid_responses_and_tokens():
    state,fake,recorder,runtime,service=setup()
    run_questions(state,service,runtime,recorder,lambda:None)
    assert fake.calls==2
    assert state['usage']['total_tokens']==26
    assert state['usage']['corrective_retries']==1
    assert state['results']['TEST_0']['status']=='processing_error'
    assert len(state['results']['TEST_0']['api_attempts'])==2
    assert all(r['response']['content']=='{' for r in state['raw_attempts'])
    assert 'source_id' in state['results']['TEST_0']['prepared']['context_sources'][0]


def test_rate_limit_pauses_all_remaining_issues_and_preserves_retrieval():
    state,fake,recorder,runtime,service=setup(fail=True,issues=2)
    run_questions(state,service,runtime,recorder,lambda:None)
    assert fake.calls==1 and state['run_status']=='paused'
    assert state['pending_issues']==['TEST_0','TEST_1']
    assert state['results']['TEST_0']['prepared']
    assert len(state['raw_attempts'])==1 and state['raw_attempts'][0]['error']['type']=='RateLimitError'


def test_completed_or_validation_failed_issue_not_regenerated_on_resume():
    state,fake,recorder,runtime,service=setup()
    state['results']['TEST_0']={'status':'partial_information','result':{'answer':'Сохранённый ответ'}}
    run_questions(state,service,runtime,recorder,lambda:None)
    assert fake.calls==0 and service.retriever.queries==[]
    assert state['results']['TEST_0']['result']['answer']=='Сохранённый ответ'


def test_failed_validation_report_does_not_publish_raw_answer_as_validated():
    state,fake,recorder,runtime,service=setup()
    run_questions(state,service,runtime,recorder,lambda:None)
    report=render_report(state)
    assert 'Проверенный итоговый ответ отсутствует' in report
    assert 'Сырой ответ API, попытка 2' in report


def test_default_cli_cannot_spend_tokens(monkeypatch):
    monkeypatch.setattr('sys.argv',['p2_answers','--persist-dir','missing'])
    with pytest.raises(SystemExit) as exc: main()
    assert exc.value.code==2
