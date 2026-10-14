"""Explicit, paid text-only control experiment. Never invoked by pytest."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from dataclasses import asdict, replace
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from app.meeting_analysis import _answer_status
from app.progress import save_checkpoint
from llm.base import LLMClient, LLMError
from llm.config import QAConfig
from llm.models import CitationValidationError, PreparedQuestion, SourceMetadataError
from llm.runtime import RunLLMClient, pause_message
from llm.service import QuestionAnsweringService
from rag.config import PROJECT_ROOT, RAGConfig
from rag.retriever import IndexNotReadyError

BASELINE=PROJECT_ROOT/'data/results/construction_meeting_20261008_174708_218282_analysis.json'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(analysis: Path, persist_dir: Path) -> dict:
    """Allow-listed configuration/files only; never environment or credentials."""
    names=['llm/service.py','llm/models.py','llm/prompts.py','llm/evidence.py',
           'llm/runtime.py','llm/config.py','llm/gigachat_client.py','app/meeting_analysis.py',
           'app/reporting.py','rag/query.py','rag/selection.py','rag/reranker.py',
           'rag/context.py','rag/retriever.py','rag/store.py','rag/embeddings.py',
           'evaluation/p2_answers.py','requirements.txt']
    paths=[PROJECT_ROOT/n for n in names]
    baseline=json.loads(analysis.read_text(encoding='utf8'))
    paths += [analysis,Path(baseline['checkpoint_file']),analysis.with_name(analysis.name.replace('_analysis.json','_report.md'))]
    paths += list((PROJECT_ROOT/'data/documents').glob('*.pdf'))
    paths += [PROJECT_ROOT/'data/documents/sources.json',PROJECT_ROOT/'evaluation/datasets/construction_meeting.json']
    paths += [p for p in persist_dir.rglob('*') if p.is_file()]
    return {'created_at':datetime.now().astimezone().isoformat(),'python':platform.python_version(),
            'packages':{n:version(n) for n in ('gigachat','httpx','sentence-transformers','chromadb','torch','PyMuPDF')},
            'files':{str(p.resolve()):digest(p) for p in paths}}


class RecordingClient(LLMClient):
    """Record the existing provider contract, including rejected generations."""
    def __init__(self,client,state,save):
        self.client,self.state,self.save=client,state,save
        self.operation=''

    @property
    def model(self): return self.client.model

    @property
    def provider(self): return self.client.provider

    def complete(self,**kwargs):
        record={'operation':self.operation,'request':kwargs,'response':None,'error':None}
        self.state['raw_attempts'].append(record)
        self.save()
        started=perf_counter()
        try:
            completion=self.client.complete(**kwargs)
            record['response']=asdict(completion)
            return completion
        except LLMError as exc:
            if getattr(exc,'completion',None) is not None:
                record['response']=asdict(exc.completion)
            record['error']={'type':type(exc).__name__,'message':str(exc),
                             'diagnostic':asdict(exc.diagnostic) if exc.diagnostic else None}
            raise
        finally:
            record['seconds']=perf_counter()-started
            self.save()


def run_questions(state, service, runtime, recorder, save):
    """Use fresh P2 prepare, never the baseline prepared dictionary."""
    state['run_status']='running'
    started=perf_counter()
    try:
        for issue in state['inputs']:
            iid=issue['id']
            row=state['results'].setdefault(iid,{'status':'pending'})
            if row['status']!='pending': continue
            recorder.operation=iid
            print(iid, 'подготовка/генерация',flush=True)
            try:
                if 'prepared' not in row:
                    row['prepared']=asdict(service.prepare(issue['normalized_question'],search_terms=issue['search_terms']))
                    save()
                prepared=PreparedQuestion.from_dict(row['prepared'])
                answer=service.answer(issue['normalized_question'],search_terms=issue['search_terms'],
                                      prepared=prepared,operation_id=iid)
                row['result']=answer.to_dict()
                row['status']=_answer_status(insufficient=answer.insufficient_context,partial=answer.partial_context,
                    answer=answer.answer,has_sources=bool(answer.cited_sources),validation=answer.validation)
            except LLMError as exc:
                row['error']={'type':type(exc).__name__,'message':pause_message(exc),
                              'diagnostic':asdict(exc.diagnostic) if exc.diagnostic else None}
                state['run_status']='paused';state['stop_reason']=row['error']
                break
            except (CitationValidationError,SourceMetadataError,ValueError) as exc:
                row['status']='processing_error'
                row['error']={'type':type(exc).__name__,'message':str(exc),
                              'diagnostics':getattr(exc,'diagnostics',None)}
            except IndexNotReadyError as exc:
                row['error']={'type':type(exc).__name__,'message':str(exc)}
                state['run_status']='paused';state['stop_reason']=row['error'];break
            finally:
                row['api_attempts']=[r for r in state['raw_attempts'] if r['operation']==iid]
                row['generation_seconds']=sum(r.get('seconds',0) for r in row['api_attempts'])
                save()
        else:
            state['run_status']='completed'
    finally:
        state['elapsed_seconds']=state.get('elapsed_seconds',0)+perf_counter()-started
        state['usage']=runtime.metrics()
        state['pending_issues']=[i['id'] for i in state['inputs'] if state['results'].get(i['id'],{}).get('status','pending')=='pending']
        save()


def render_report(state: dict) -> str:
    lines=['# P2: текстовый контроль нормативных ответов','',
           f"Состояние: {state['run_status']}. Модель: {state['qa_config']['model']}.",'',
           'Сырой ответ API и опубликованный ответ после P0 показаны отдельно. Успех проверки происхождения не удостоверяет применимость нормы.','']
    for issue in state['inputs']:
        row=state['results'].get(issue['id'],{'status':'pending'})
        lines += [f"## {issue['id']}",'',issue['normalized_question'],'',
                  f"Статус: {row['status']}",'']
        if 'error' in row: lines += ['Ошибка:',json.dumps(row['error'],ensure_ascii=False),'']
        result=row.get('result')
        if result: lines += ['### Итог программы','',result['answer'],'']
        for n,attempt in enumerate(row.get('api_attempts',[]),1):
            lines += [f'### Сырой ответ API, попытка {n}','',
                      '```json',(attempt.get('response') or {}).get('content') or 'Ответ не получен.','```','']
        cited=result.get('cited_sources',[]) if result else []
        for source in cited:
            lines += [f"### [{source['source_id']}] {source['designation']} · PDF с. {source['page']}",
                      f"Chunk: `{source['chunk_id']}`",'',source['text'],'']
        if not result: lines += ['Проверенный итоговый ответ отсутствует; сырая генерация не заменяет его.','']
    lines += ['## Учёт API','', '```json',json.dumps(state.get('usage',{}),ensure_ascii=False,indent=2),'```','']
    return '\n'.join(lines)


def main() -> int:
    parser=argparse.ArgumentParser(description='ЯВНЫЙ ПЛАТНЫЙ текстовый контроль 8 вопросов, без аудио/detection.')
    parser.add_argument('--analysis',type=Path,default=BASELINE)
    parser.add_argument('--persist-dir',type=Path,required=True,help='Copy of existing index; no reindex')
    parser.add_argument('--output-dir',type=Path,help='New directory, must not exist')
    parser.add_argument('--resume',type=Path,help='P2 results.json only, not historical baseline progress')
    parser.add_argument('--allow-api',action='store_true',help='Explicitly authorize paid generation')
    args=parser.parse_args()
    if not args.allow_api: parser.error('Нужен явный --allow-api. По умолчанию API не вызывается.')
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT/'.env',override=False)
    config=QAConfig.from_env()
    if config.model!='GigaChat-3-Ultra': parser.error('Для P2 нужна существующая модель GigaChat-3-Ultra; настройки не менялись.')
    if args.resume:
        output=args.resume.resolve().parent
        state=json.loads(args.resume.read_text(encoding='utf8'))
        if state.get('experiment')!='P2' or state['qa_config']!=asdict(config):
            parser.error('Несовместимый P2 checkpoint/config')
        if state['fingerprint']!=freeze(args.analysis,args.persist_dir)['files']:
            parser.error('Код или входные файлы изменились; продолжение эксперимента запрещено')
    else:
        output=(args.output_dir or PROJECT_ROOT/'data/results'/('P2_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
        output.mkdir(parents=True,exist_ok=False)
        frozen=freeze(args.analysis,args.persist_dir)
        save_checkpoint(output/'manifest.json',frozen)
        baseline=json.loads(args.analysis.read_text(encoding='utf8'))
        inputs=[{k:i[k] for k in ('id','normalized_question','search_terms','original_text','segment_ids','start','end')}
                for i in baseline['detected_issues']]
        if len(inputs)!=8 or len({i['id'] for i in inputs})!=8: parser.error('Для P2 ожидаются восемь уникальных сохранённых вопросов')
        state={'experiment':'P2','qa_config':asdict(config),'fingerprint':frozen['files'],
               'inputs':inputs,'baseline_analysis':str(args.analysis.resolve()),
               'results':{},'raw_attempts':[],'llm_runtime':{},'elapsed_seconds':0,'run_status':'created'}
        save_checkpoint(output/'inputs.json',{'issues':inputs})
    def save():
        save_checkpoint(output/'results.json',state)
        (output/'report.md').write_text(render_report(state),encoding='utf8')
    save()
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['ANONYMIZED_TELEMETRY']='False'
    from llm.gigachat_client import GigaChatClient
    from rag.retriever import SemanticRetriever
    client=None
    try:
        client=GigaChatClient.from_env(model=config.model,timeout_seconds=config.request_timeout_seconds)
        recorder=RecordingClient(client,state,save)
        runtime=RunLLMClient(recorder,state=state['llm_runtime'],
            max_calls=min(16,int(os.getenv('MAX_LLM_CALLS_PER_RUN','16'))),on_change=save)
        service=QuestionAnsweringService(retriever=SemanticRetriever(replace(RAGConfig(),persist_dir=args.persist_dir)),llm_client=runtime,config=config)
        run_questions(state,service,runtime,recorder,save)
        print('Результаты:',output)
        print('Учёт API:',json.dumps(state['usage'],ensure_ascii=False))
        return 0 if state['run_status']=='completed' else 1
    except LLMError as exc:
        state['run_status']='paused';state['stop_reason']={'type':type(exc).__name__,'message':str(exc)};save()
        print('Запуск приостановлен:',str(exc));return 1
    finally:
        if client is not None: client.close()


if __name__=='__main__': raise SystemExit(main())
