"""P1.1 synthetic selectors: no control IDs, models, API or ground truth."""
from dataclasses import asdict, replace
import pytest
from llm.config import QAConfig
from llm.models import PreparedQuestion, SourceEvidence
from llm.service import QuestionAnsweringService
from rag.selection import aspects, classify
from rag.selection import mandatory_link
from rag.test_p1_retrieval import hit, Retriever


def source(text, index, rank=.8, **kwargs):
    return replace(SourceEvidence.from_search_result(index, hit(text,index),rerank_score=rank), **kwargs)


def choose(rows, question, terms=(), **config):
    qa=QuestionAnsweringService(retriever=Retriever([]),llm_client=None,config=QAConfig(**config))
    qa._retrieval_trace={'records':[{'source_id':s.source_id,'decision':'candidate'} for s in rows]}
    return qa._fit_context(rows,question,terms), qa._retrieval_trace


def test_direct_requirement_beats_general_explanation():
    broad=source('Ограждения галереи должны обеспечивать безопасность людей.',1,.95)
    direct=source('Высота ограждения галереи должна быть не менее 1,4 м.',2,.75)
    chosen,_=choose([broad,direct],'Высота ограждения галереи?', ['высота ограждения галереи'],top_k=1)
    assert chosen==[direct]


def test_explanatory_link_does_not_displace_an_independent_direct_requirement():
    parent=source('Температура склада должна быть не более 20 градусов. См. описание по 3.1.',1)
    link=source('3.1 Описание склада и его назначения.',2,related_to=parent.chunk_id,
                provenance=[{'method':'related','reference':'3.1'}])
    independent=source('Влажность склада должна быть не более 65 процентов.',3,.7)
    chosen,_=choose([parent,link,independent],'Температура и влажность склада?',
                    ['температура склада','влажность склада'],top_k=2)
    assert independent in chosen and link not in chosen


def test_separate_parameters_are_covered_before_redundant_requirement():
    rows=[source('Температура склада должна быть не более 20 градусов.',1,.9),
          source('Температура склада должна быть не менее 15 градусов.',2,.89),
          source('Влажность склада должна быть не более 65 процентов.',3,.7)]
    chosen,trace=choose(rows,'Температура и влажность склада?',
                        ['температура склада','влажность склада'],top_k=2)
    assert rows[2] in chosen and len(chosen)==2
    assert not trace['uncovered_aspects']
    assert trace['completeness_verified'] is False


def test_required_exception_has_reserved_slot_and_original_text():
    parent=source('Толщина покрытия резервуара должна определяться по 4.1.2.',1,.9)
    linked=source('4.1.2 Исключение применяется только при капитальном ремонте.',2,
                  related_to=parent.chunk_id,provenance=[{'method':'related','reference':'4.1.2'}])
    other=source('Температура резервуара должна быть не более 20 градусов.',3,.8)
    chosen,trace=choose([parent,linked,other],'Толщина покрытия и температура резервуара?',
                        ['толщина покрытия резервуара','температура резервуара'],top_k=2)
    assert chosen==[parent,linked]
    assert any(r['selection']['role']=='mandatory_condition' for r in trace['records'])
    assert trace['uncovered_aspects']


def test_whole_chunk_budget_records_uncovered_and_missing_dependency():
    parent=source('Толщина покрытия резервуара должна определяться по 4.1.2.',1)
    linked=source('4.1.2 Исключение. '+'При капитальном ремонте. '*100,2,
                  related_to=parent.chunk_id,provenance=[{'method':'related','reference':'4.1.2'}])
    chosen,trace=choose([parent,linked],'Толщина покрытия резервуара?',
                        ['толщина покрытия резервуара'],max_context_chars=1000)
    assert chosen==[]
    assert trace['records'][0]['decision']=='dependency_budget'
    assert trace['uncovered_aspects']


def test_absence_of_direct_requirement_is_not_claimed_as_coverage():
    rows=[source('Галерея должна обеспечивать безопасность людей.',1)]
    chosen,trace=choose(rows,'Высота ограждения галереи?', ['высота ограждения галереи'])
    assert chosen==rows
    assert trace['uncovered_aspects']


def test_existing_union_beyond_twenty_is_available_without_deeper_search():
    class Union(Retriever):
        def retrieve(self,query,top_k):
            self.queries.append((query,top_k))
            base=0 if len(self.queries)==1 else 20
            return [hit(f'Общие положения галереи {i}.',i,.95) for i in range(base,base+20)]
    qa=QuestionAnsweringService(retriever=Union([]),llm_client=None,config=QAConfig())
    prepared=qa.prepare('Галерея?',search_terms=['ограждение'])
    assert len(prepared.retrieved_candidates)==40
    assert all(k==20 for _,k in qa.retriever.queries)
    assert prepared.retrieval_trace['internal_pool_depth']==40
    assert prepared.retrieval_trace['records'][25]['source_id']=='SOURCE_26'


def test_source_ids_and_provenance_survive_reordered_context_and_roundtrip():
    broad=source('Галерея должна обеспечивать безопасность.',1,.9,provenance=[{'query':'галерея','method':'lexical'}])
    direct=source('Высота ограждения галереи должна быть 1,4 м.',21,.7,provenance=[{'query':'высота ограждения','method':'semantic'}])
    chosen,trace=choose([broad,direct],'Высота ограждения галереи?', ['высота ограждения галереи'])
    p=PreparedQuestion('Вопрос','Вопрос',[broad,direct],chosen,.1,trace)
    assert chosen[0].source_id=='SOURCE_21'
    assert PreparedQuestion.from_dict(asdict(p))==p
    assert chosen[0].provenance==direct.provenance


def test_single_correct_question_is_not_regressed():
    direct=source('Высота ограждения галереи должна быть 1,4 м.',1)
    chosen,trace=choose([direct],'Высота ограждения галереи?', ['высота ограждения галереи'])
    assert chosen==[direct] and not trace['uncovered_aspects']


def test_spatial_conflict_is_a_diagnostic_not_legal_applicability():
    requested=aspects('Ширина пандуса у входа?', ['ширина пандуса'])
    role=classify(source('Ширина пандуса внутри здания должна быть 1,2 м.',1),requested)
    assert role['scope_conflicts'] and not role['direct_aspects']
    assert role['applicability_verified'] is False


def test_parameter_of_other_object_cannot_cover_requested_dimension():
    wrong=source('Здания высотой четыре этажа должны иметь эвакуационные выходы.',1)
    requested=aspects('Высота эвакуационного выхода?', ['высота эвакуационного выхода'])
    assert not classify(wrong,requested)['direct_aspects']


def test_object_inside_condition_is_not_main_normative_requirement():
    wrong=source('Допускается размещение сидений при наличии аварийного освещения.',1)
    requested=aspects('Требуется ли аварийное освещение?', ['аварийное освещение'])
    assert not classify(wrong,requested)['direct_aspects']


def test_link_in_condition_list_inherits_heading_not_neighbouring_clause():
    parent=source('4.1.1 Вентиляцию кухни допускается объединять при следующих условиях: '
                  '- применения индивидуальной установки; - устройства канала по 4.2.3.',1)
    child=source('4.2.3 Канал должен быть отдельным.',2,provenance=[{'method':'related','reference':'4.2.3'}])
    assert mandatory_link(parent,child,aspects('Вентиляция кухни?', ['вентиляция кухни']))
    unrelated=replace(parent,text='4.1.1 Вентиляция кухни должна обеспечивать воздухообмен. '
                      '4.1.2 Покрытие галереи следует выполнять по 4.2.3.')
    assert not mandatory_link(unrelated,child,aspects('Вентиляция кухни?', ['вентиляция кухни']))


def test_direct_requirement_beyond_twenty_can_reach_context(monkeypatch):
    from rag.reranker import RerankedResult
    rows=[hit('Общие положения галереи должны обеспечивать безопасность.',i,.9) for i in range(40)]
    rows[25]=hit('Высота ограждения галереи должна быть 1,4 м.',25,.9)
    class Union(Retriever):
        def retrieve(self,query,top_k):
            self.queries.append(query)
            return rows[:20] if len(self.queries)==1 else rows[20:]
    # Hold upstream ranking fixed: test context selection, not rerank quality.
    monkeypatch.setattr('llm.service.rerank_candidates',lambda q,t,c,top_k:
                        [RerankedResult(r,1-i*.01,()) for i,r in enumerate(c)])
    qa=QuestionAnsweringService(retriever=Union([]),llm_client=None,config=QAConfig(top_k=1))
    p=qa.prepare('Высота ограждения галереи?',search_terms=['ограждение галереи'])
    assert p.context_sources[0].source_id=='SOURCE_26'
    assert p.context_sources[0].text==rows[25].text
    assert p.context_sources[0].provenance


def test_parent_is_not_added_without_room_for_its_required_condition():
    parent=source('Толщина покрытия резервуара должна определяться по 4.1.2.',1)
    linked=source('4.1.2 Исключение при капитальном ремонте.',2,
                  related_to=parent.chunk_id,provenance=[{'method':'related','reference':'4.1.2'}])
    chosen,trace=choose([parent,linked],'Толщина покрытия резервуара?', ['толщина покрытия резервуара'],top_k=1)
    assert chosen==[]
    assert trace['records'][0]['decision']=='dependency_slots'
