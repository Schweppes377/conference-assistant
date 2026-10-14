"""Explicit local retrieval experiment, never a default pytest/API operation.

The control dataset is read only after production parameters are fixed. Its
sources are used for scoring, never query preparation or reranking.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import socket
from dataclasses import asdict,replace
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from evaluation.metrics import detection_metrics,retrieval_metrics
from llm.config import QAConfig
from llm.service import QuestionAnsweringService
from rag.config import RAGConfig
from rag.context import chunk_key
from rag.reranker import duplicate,chunk_kind,VERSION,WEIGHTS
from rag.models import SearchResult
from rag.selection import VERSION as SELECTION_VERSION

ROOT=Path(__file__).resolve().parents[1]

def diagnostics(sources: list[dict]) -> dict:
    rows=[SearchResult(s['distance'],s['text'],{'source_file':s['chunk_id'].split(':p')[0]}) for s in sources]
    pairs=sum(duplicate(a,b) for index,a in enumerate(rows) for b in rows[index+1:])
    service=sum(chunk_kind(s['text']) in {'publication','contents','references','title'} for s in sources)
    return {'chunks':len(sources),'duplicate_pairs':pairs,'service_fragments':service,
            'service_fraction':service/len(sources) if sources else None,
            'method':'P1 lexical duplicate/service heuristics, same detector applied to before and after; not expert relevance labels'}

def score_lists(dataset: dict,analysis: dict,lists: dict) -> dict:
    refs=dataset.get('reference_issues')
    if dataset.get('verification',{}).get('status')!='verified' or not refs:
        return retrieval_metrics([])
    detected=detection_metrics(refs,analysis['detected_issues'],**dataset.get('matching',{}))
    pairs=dict(detected.get('matches',[]))
    queries=[]
    for index,ref in enumerate(refs):
        matched=analysis['detected_issues'][pairs[index]]['id'] if index in pairs else None
        sources=lists.get(matched)
        queries.append({'reference_id':ref['id'],'expected':ref.get('expected_sources'),
                        'sources':sources or [],'available':sources is not None,'matched_prediction_id':matched})
    return retrieval_metrics(queries)

def compare(analysis: dict,progress: dict,dataset: dict,retriever) -> dict:
    service=QuestionAnsweringService(retriever=retriever,llm_client=None,config=QAConfig(**progress['settings']['qa']))
    after,observations={},[]
    started=perf_counter()
    for issue in analysis['detected_issues']:
        prepared=service.prepare(issue['normalized_question'],search_terms=issue.get('search_terms',[]))
        after[issue['id']]=asdict(prepared)
        old=progress['prepared'][issue['id']]
        observations.append({'id':issue['id'],'question':issue['normalized_question'],
            'baseline_retrieval_sec':old['retrieval_seconds'],'p1_retrieval_sec':prepared.retrieval_seconds,
            'baseline_candidates':[s['chunk_id'] for s in old['retrieved_candidates']],
            'p1_candidates':[s.chunk_id for s in prepared.retrieved_candidates],
            'baseline_context':[s['chunk_id'] for s in old['context_sources']],
            'p1_context':[s.chunk_id for s in prepared.context_sources],
            'baseline_diagnostics':diagnostics(old['context_sources']),
            'p1_diagnostics':diagnostics(after[issue['id']]['context_sources']),
            'query_count':len(prepared.retrieval_trace['queries']),
            'context_characters':prepared.retrieval_trace['context_characters']})
    elapsed=perf_counter()-started
    # Reference scoring follows all searches; not accessible to production.
    before_candidates={i['id']:progress['prepared'][i['id']]['retrieved_candidates'] for i in analysis['detected_issues']}
    before_context={i['id']:progress['prepared'][i['id']]['context_sources'] for i in analysis['detected_issues']}
    candidate_after={i:p['retrieved_candidates'] for i,p in after.items()}
    context_after={i:p['context_sources'] for i,p in after.items()}
    # Independently prepared indexed-text hashes are checked against the same
    # existing corpus. This attests identity, not current legal applicability.
    corpus=retriever._get_store().corpus()[0]
    by_key={chunk_key(s):s for s in corpus}
    ref_checks=[]
    for ref in dataset.get('reference_issues',[]):
        for expected in ref.get('expected_sources',[]):
            row=by_key.get(expected.get('chunk_id'))
            stored_hash=hashlib.sha256(row.text.encode('utf-8')).hexdigest() if row else None
            ref_checks.append({'reference_id':ref['id'],'chunk_id':expected.get('chunk_id'),
                'exists':row is not None,'hash_matches':stored_hash==expected.get('indexed_text_sha256')})
    verified=dataset.get('verification',{}).get('status')=='verified' and bool(ref_checks) and all(r['exists'] and r['hash_matches'] for r in ref_checks)
    scoring_dataset=dataset if verified else {'verification':{'status':'pending'}}
    return {'algorithm_version':VERSION,'selection_version':SELECTION_VERSION,'weights':WEIGHTS,'api_calls':0,'audio_runs':0,
            'reference_verification':dataset.get('verification'),'indexed_reference_checks':ref_checks,
            'metrics_available':verified,'reference_limitations':'pre-existing source labels, not an independent expert judgement of newly retrieved alternatives; exact identity scoring excludes unlisted equivalent clauses',
            'baseline_candidates':score_lists(scoring_dataset,analysis,before_candidates),
            'baseline_context':score_lists(scoring_dataset,analysis,before_context),
            'p1_candidates':score_lists(scoring_dataset,analysis,candidate_after),
            'p1_context':score_lists(scoring_dataset,analysis,context_after),
            'baseline_historical_retrieval_sec':sum(o['baseline_retrieval_sec'] for o in observations),
            'p1_warm_retrieval_sec':sum(o['p1_retrieval_sec'] for o in observations),
            'comparison_loop_sec':elapsed,'observations':observations,'p1_prepared':after}

def main() -> int:
    parser=argparse.ArgumentParser(description='Локальное сравнение поиска P1 без LLM/ASR и без изменения эталона.')
    parser.add_argument('--analysis',type=Path,default=ROOT/'data/results/construction_meeting_20261008_174708_218282_analysis.json')
    parser.add_argument('--dataset',type=Path,default=ROOT/'evaluation/datasets/construction_meeting.json')
    parser.add_argument('--persist-dir',type=Path,required=True,help='Existing index or a read-only-source snapshot; no reindex')
    parser.add_argument('--output',type=Path,required=True,help='New experiment JSON; existing output will not be overwritten')
    parser.add_argument('--baseline-retrieval',type=Path,help='Saved P1 experiment; scoring only, never context selection')
    args=parser.parse_args()
    if args.output.exists(): parser.error('Укажите новый файл результата, исторические измерения не перезаписываются')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['ANONYMIZED_TELEMETRY']='False'
    with patch.object(socket.socket,'connect',side_effect=RuntimeError('Evaluation network access forbidden')):
        from rag.retriever import SemanticRetriever
        analysis=json.loads(args.analysis.read_text(encoding='utf-8'))
        progress=json.loads(Path(analysis['checkpoint_file']).read_text(encoding='utf-8'))
        retriever=SemanticRetriever(replace(RAGConfig(),persist_dir=args.persist_dir))
        initialization=perf_counter()
        retriever._get_store().corpus()
        retriever._get_embedder().encode_query('Контроль подготовки поиска')
        initialization=perf_counter()-initialization
        dataset=json.loads(args.dataset.read_text(encoding='utf-8'))
        result=compare(analysis,progress,dataset,retriever)
        if args.baseline_retrieval:
            baseline=json.loads(args.baseline_retrieval.read_text(encoding='utf-8'))
            if baseline['artifacts'] != [{'file':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                                        for p in (args.analysis,args.dataset,Path(analysis['checkpoint_file']))]:
                raise ValueError('Baseline inputs differ; cannot compare matched experiments')
            result['previous_p1']={k:baseline[k] for k in ('p1_candidates','p1_context','p1_warm_retrieval_sec','initialization_sec')}
            result['previous_p1']['sha256']=hashlib.sha256(args.baseline_retrieval.read_bytes()).hexdigest()
        result['initialization_sec']=initialization
        result['device']=retriever.device
        result['index_count']=retriever.index_count
        result['artifacts']=[{'file':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in (args.analysis,args.dataset,Path(analysis['checkpoint_file']))]
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print('Результат:',args.output)
        for name in ('baseline_candidates','baseline_context','p1_candidates','p1_context'):
            print(name,{k:result[name].get(k) for k in ('hit_at_1','hit_at_3','hit_at_5','mrr')})
        print('Время поиска (прогрето):',result['p1_warm_retrieval_sec'],'с; подготовка:',initialization,'с')
    return 0

if __name__=='__main__': raise SystemExit(main())
