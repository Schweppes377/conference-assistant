"""No real index/model: verify comparison doesn't invent reference metrics."""
from evaluation.p1_compare import score_lists,diagnostics

def test_unverified_reference_metrics_are_unavailable():
    result=score_lists({'verification':{'status':'pending'}},{'detected_issues':[]},{})
    assert result['hit_at_1'] is None and result['mrr'] is None

def test_existing_metrics_used_without_changing_matching_or_source_identity():
    source={'document':'Synthetic','designation':'СП TEST','page':1,'chunk_id':'synthetic:p1:c0'}
    dataset={'verification':{'status':'verified'},'matching':{'mode':'time','threshold':.5},
             'reference_issues':[{'id':'REF_TEST','start':0,'end':3,'expected_sources':[source]}]}
    analysis={'detected_issues':[{'id':'TEST','start':0,'end':3}]}
    result=score_lists(dataset,analysis,{'TEST':[{'document':'Wrong'},source]})
    assert result['hit_at_1']==0 and result['hit_at_3']==1 and result['mrr']==.5

def test_service_diagnostics_are_explicitly_heuristic():
    source={'text':'Ключевые слова: галерея. Редактор.','distance':.1,'chunk_id':'synthetic:p1:c0'}
    result=diagnostics([source])
    assert result['service_fragments']==1
    assert 'not expert relevance labels' in result['method']
