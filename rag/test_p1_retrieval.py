"""Synthetic training/regression cases; no control meeting IDs or real models."""
from dataclasses import asdict
import pytest

from rag.query import prepare_queries,tokens
from rag.reranker import rerank_candidates,duplicate,chunk_kind
from rag.context import related_chunks
from rag.models import SearchResult,TextChunk
from rag.store import VectorStore
from llm.service import QuestionAnsweringService
from llm.models import SourceEvidence,PreparedQuestion
from llm.config import QAConfig

def hit(text,index=0,score=.9,file='synthetic.pdf',page=1):
    return SearchResult(1-score,text,{'source_file':file,'page':page,'chunk_index':index,'title':'Синтетический норматив','designation':'СП TEST'})

class Retriever:
    def __init__(self,rows): self.rows=rows;self.queries=[]
    def retrieve(self,query,top_k): self.queries.append(query);return self.rows[:top_k]

def service(rows,**kwargs):
    return QuestionAnsweringService(retriever=Retriever(rows),llm_client=None,config=QAConfig(**kwargs))

def test_dialog_removed_but_technical_phrase_preserved():
    plan=prepare_queries('Какая высота ограждения галереи?', ['И еще ограждение','Вот это уже хороший вопрос','Насколько я понимаю','высота ограждения галереи'])
    assert not any('хороший вопрос' in q or 'Насколько' in q or 'И еще' in q for q in plan.queries)
    assert 'высота ограждения галереи' in plan.queries
    assert len(plan.queries)<=8

def test_bare_parameter_is_contextualized_when_object_is_available():
    plan=prepare_queries('Толщина защитного покрытия резервуара?', ['защитное покрытие резервуара','толщина'])
    assert 'толщина' not in plan.queries
    assert any('толщина' in q and 'резервуара' in q for q in plan.queries)
    assert 'толщина' in plan.terms

def test_referenced_conditions_get_slot_after_parent_without_new_queries():
    parent=hit('Толщину покрытия резервуара следует выбирать по 4.1.2.',0,.95)
    other=hit('Толщина резервуара должна соответствовать расчёту.',1,.94)
    linked=hit('4.1.2 Исключение применяется только при капитальном ремонте.',2,.50)
    class Linked(Retriever):
        def related(self,p,limit=2):
            return [SearchResult(.50,r.text,{**r.metadata,'_score_inherited':False})
                    for r in related_chunks(p,[parent,linked],limit=limit)]
    qa=service([parent,other],top_k=2);qa.retriever=Linked([parent,other])
    prepared=qa.prepare('Толщина покрытия резервуара?',search_terms=['покрытие резервуара','толщина'])
    assert parent.text in [s.text for s in prepared.context_sources]
    assert linked.text in [s.text for s in prepared.context_sources]
    assert next(s for s in prepared.context_sources if s.text==linked.text).score==.50
    assert any(r['decision']=='included_related' for r in prepared.retrieval_trace['records'])
    assert len(qa.retriever.queries)==len(prepared.retrieval_trace['queries'])

def test_unselectable_parent_cannot_consume_link_budget():
    weak=hit('Толщина покрытия следует по 4.1.2.',0,.4)
    valid=hit('Толщина покрытия резервуара должна быть 2 мм.',1,.9)
    class Linked(Retriever):
        def related(self,p,limit=2):
            assert p.score>=.84
            return []
    qa=service([weak,valid]);qa.retriever=Linked([weak,valid])
    assert qa.prepare('Толщина покрытия резервуара?').context_sources

def test_russian_inflections_do_not_lose_key_parameter():
    assert tokens('высота ограждения')==tokens('высоту ограждений')
    assert tokens('вентиляция')==tokens('вентиляции')
    assert tokens('освещение')==tokens('освещения')
    assert tokens('эвакуационный')==tokens('эвакуация')
    assert tokens('площадь')==tokens('площади')

def test_specific_object_and_parameter_beats_general_text():
    q='Какова высота ограждения открытой грузовой галереи?'
    terms=['ограждение открытой грузовой галереи','высота ограждения']
    general=hit('Ограждения должны обеспечивать безопасность людей.',0,.95)
    specific=hit('Высота ограждения открытой грузовой галереи должна быть не менее 1,4 м.',1,.87)
    assert rerank_candidates(q,terms,[general,specific],top_k=2)[0].result==specific

def test_long_repetition_does_not_earn_density_bonus():
    text='Высота ограждения грузовой галереи должна быть 1,4 м.'
    short,long=hit(text),hit((text+'\n\n')*30,1)
    ranked=rerank_candidates('Высота ограждения грузовой галереи?',['высота ограждения'],[short,long],top_k=2)
    assert ranked[0].rerank_score==ranked[1].rerank_score

def test_qualified_object_not_replaced_by_shared_generic_words():
    q='Какие требования к антикоррозионному покрытию галереи?'
    terms=['антикоррозионное покрытие','галерея']
    matching=hit('Антикоррозионное покрытие следует предусматривать.',0,.9)
    generic=hit('Покрытие галереи следует предусматривать для движения людей.',1,.9)
    ranked=rerank_candidates(q,terms,[generic,matching],top_k=2)
    assert ranked[0].result==matching

def test_context_starts_with_highest_eligible_relevance():
    rows=[hit('Антикоррозионное покрытие должно защищать резервуар.',0,.95),
          hit('Покрытие резервуара должно обеспечивать герметичность.',1,.85)]
    prepared=service(rows).prepare('Антикоррозионное покрытие резервуара?',search_terms=['антикоррозионное покрытие'])
    assert prepared.context_sources[0].source_id==prepared.retrieved_candidates[0].source_id

@pytest.mark.parametrize('text',['Содержание\n1 Общие положения .... 5\n2 Галереи .... 6\n3 Требования .... 8',
    'Ключевые слова: галерея, ограждение, высота. Редактор Иванов.',
    'Нормативные ссылки\nГОСТ 100\nСП 200', 'Министерство. Утвержден. Дата введения.'])
def test_service_pages_cannot_fill_context(text):
    prepared=service([hit(text,0,.99),hit('Высота ограждения галереи должна быть 1,4 м.',1,.86)]).prepare('Высота ограждения галереи?')
    assert len(prepared.context_sources)==1
    assert 'должна' in prepared.context_sources[0].text
    assert any(r['decision']=='service_fragment' for r in prepared.retrieval_trace['records'])

def test_definitions_are_retained_when_requested():
    definition=hit('Термины и определения. 3.1 галерея: сооружение для движения людей.',0,.94)
    requirement=hit('Галерея должна обеспечивать проход.',1,.9)
    assert chunk_kind(definition.text)=='definition'
    assert rerank_candidates('Что означает термин галерея?',['галерея'],[definition,requirement],top_k=2)[0].signals['service_penalty']==0

def test_deduplication_keeps_all_query_origins():
    prepared=service([hit('Температура помещения должна поддерживаться.'),hit('Температура помещения должна поддерживаться.')]).prepare('Температура помещения?',search_terms=['помещение','температура'])
    assert len(prepared.retrieved_candidates)==1
    assert len(prepared.retrieved_candidates[0].provenance)>=2
    assert all('query' in h and 'initial_position' in h for h in prepared.retrieved_candidates[0].provenance)

def test_duplicate_policy_preserves_differing_numbers_and_exceptions():
    assert duplicate(hit('Высота 1,4 м.'),hit('Высота 1,4 м.'))
    assert not duplicate(hit('Высота 1,4 м.'),hit('Высота 1,5 м.'))
    assert not duplicate(hit('Высота 1,4 м.'),hit('Высота 1,4 м, кроме ремонтируемых объектов.'))

def test_compound_aspects_and_diversity():
    rows=[hit('Температура склада должна поддерживаться по расчёту.',0),
          hit('Температура склада должна поддерживаться по расчёту.',1),
          hit('Влажность склада должна быть не более 65 процентов.',2,.88,'other.pdf')]
    prepared=service(rows,top_k=2).prepare('Какие температура и влажность склада?',search_terms=['температура склада','влажность склада'])
    assert len(prepared.context_sources)==2
    assert any('Влажность' in s.text for s in prepared.context_sources)
    assert any(r['decision']=='duplicate' for r in prepared.retrieval_trace['records'])

def test_skip_oversized_candidate_and_continue():
    prepared=service([hit('Требования к галерее. '*100,0,.99),hit('Галерея должна иметь ограждение.',1,.9)],max_context_chars=1000).prepare('Галерея?')
    assert len(prepared.context_sources)==1
    assert any(r['decision']=='context_budget' for r in prepared.retrieval_trace['records'])

def test_table_and_clause_links_stay_in_same_document():
    parent=hit('Параметры следует принимать по 4.1.2 и согласно таблице 3.',0)
    foreign=hit('4.1.2 Параметры должны быть ошибочными. Таблица 3',1,file='foreign.pdf')
    clause=hit('4.1.2 Параметры должны соответствовать расчёту, кроме указанного исключения.',2,page=2)
    table=hit('Таблица 3 Высота, м\nГалерея 1,4',3,page=3)
    rows=related_chunks(parent,[parent,foreign,clause,table],limit=2)
    assert len(rows)==2 and all(r.metadata['source_file']=='synthetic.pdf' for r in rows)
    assert all(r.metadata['_related_to']=='synthetic.pdf:p1:c0' for r in rows)
    assert clause.text in [r.text for r in rows] and table.text in [r.text for r in rows]

def test_foreign_qualified_reference_not_assumed_local():
    parent=hit('Согласно СП 999 пункту 4.1.2 требуется проверить.',0)
    assert not related_chunks(parent,[hit('4.1.2 Параметры должны определяться.',1)],limit=2)

def test_linked_metadata_round_trip_and_budget():
    parent=hit('Параметры галереи следует принимать по 4.1.2.',0)
    linked=hit('4.1.2 Высота ограждения галереи должна быть 1,4 м, кроме ремонтируемых объектов.',1,page=2)
    class Linked(Retriever):
        def related(self,p,limit=2): return related_chunks(p,[parent,linked],limit=limit)
    qa=service([parent],top_k=2);qa.retriever=Linked([parent])
    prepared=qa.prepare('Высота ограждения галереи?')
    assert len(prepared.retrieved_candidates)==2
    assert prepared.retrieved_candidates[1].related_to=='synthetic.pdf:p1:c0'
    restored=PreparedQuestion.from_dict(asdict(prepared))
    assert restored==prepared
    assert 'кроме' in restored.retrieved_candidates[1].text
    assert restored.retrieval_trace['records'][1]['source_id']=='SOURCE_2'

def test_no_relevant_sources_or_empty_index_does_not_invent_context():
    assert service([]).prepare('Галерея?').context_sources==[]
    assert service([hit('Ключевые слова: ограждение.',score=.99)]).prepare('Галерея?').context_sources==[]
    with pytest.raises(ValueError): service([]).prepare(' ')

def test_lexical_candidates_have_real_cosine_scores_and_cache_invalidates(tmp_path):
    store=VectorStore(tmp_path/'db','synthetic_p1','test-model')
    chunks=[TextChunk('test.pdf','Test','СП TEST',1,0,'Температура склада должна быть стабильной.'),
            TextChunk('test.pdf','Test','СП TEST',2,1,'Влажность склада не более 65 процентов.')]
    store.replace_document('test.pdf',chunks,[[1,0],[0,1]])
    hits=store.lexical_search('Температура склада',[1,0],2)
    assert hits[0].metadata['chunk_index']==0
    assert hits[0].score==pytest.approx(1.)
    assert store.count==2
    store.replace_document('test.pdf',[],[])
    assert store.lexical_search('Температура',[1,0],2)==[]
