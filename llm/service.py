"""Grounded Question → retrieval → LLM orchestration."""

from __future__ import annotations

import re
from collections.abc import Sequence
from decimal import Decimal
from dataclasses import replace
from time import perf_counter
from typing import Protocol

from llm.base import EmptyResponseError, LLMClient
from llm.config import QAConfig
from llm.models import (
    CitationValidationError, ContextValidationError, PreparedQuestion,
    QAResult, QATimings, SourceEvidence,
)
from llm.runtime import generate
from llm.prompts import (
    INSUFFICIENT_CONTEXT_ANSWER,
    PARTIAL_CONTEXT_MARKER,
    SYSTEM_PROMPT,
    build_user_prompt,
)
from rag.models import SearchResult
from rag.reranker import rerank_candidates, chunk_kind, duplicate, VERSION
from rag.query import prepare_queries, tokens
from rag.context import chunk_key, related_chunks
from rag.selection import aspects, classify, mandatory_link, explicit_location, INTERNAL_POOL, VERSION as SELECTION_VERSION
from llm.evidence import parse_evidence, legacy_assessment, render_evidence


_SOURCE_CITATION = re.compile(r"\[\s*(SOURCE_\d+(?:\s*,\s*SOURCE_\d+)*)\s*\]")


def _citation_ids(answer: str) -> list[str]:
    remaining = _SOURCE_CITATION.sub("", answer)
    if re.search(r"SOURCE_\w+", remaining):
        raise CitationValidationError("Некорректный формат ссылки: используйте [SOURCE_N] или [SOURCE_1, SOURCE_3]")
    return list(dict.fromkeys(sid for group in _SOURCE_CITATION.findall(answer)
                              for sid in re.findall(r"SOURCE_\d+", group)))


_MEASURED_VALUE = re.compile(
    r"(?<![\w.,])(?P<value>\d+(?:[.,]\d+)?)\s*"
    r"(?P<unit>м²|м2|кв\.\s*м|квадратн(?:ый|ых|ого)\s+метр(?:а|ов)?|"
    r"сантиметр(?:а|ов)?|миллиметр(?:а|ов)?|метр(?:а|ов)?|см|мм|м|дб(?:а)?|%|процент(?:а|ов)?|"
    r"чел(?:овек(?:а|ов)?)?)(?![A-Za-zА-Яа-яЁё\d²³])",
    re.IGNORECASE,
)


class Retriever(Protocol):
    """Provider-independent retrieval contract used by the Q&A service."""

    def retrieve(self, query: str, top_k: int) -> list[SearchResult]: ...


class QuestionAnsweringService:
    """Combine existing semantic retrieval with a replaceable LLM provider."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        llm_client: LLMClient,
        config: QAConfig,
    ) -> None:
        self.retriever = retriever
        self.llm_client = llm_client
        self.config = config
        self._retrieval_trace: dict = {}

    def answer(
        self,
        question: str,
        *,
        search_terms: Sequence[str] = (),
        prepared: PreparedQuestion | None = None,
        operation_id: str = "standalone",
    ) -> QAResult:
        """Answer one question using only verified retrieved chunks."""

        clean_question = question.strip()
        if not clean_question:
            raise ValueError("Question cannot be empty")
        trusted_search_terms = [
            term.strip()
            for term in search_terms
            if isinstance(term, str) and term.strip()
        ]
        validation_input = " ".join([clean_question, *trusted_search_terms])

        total_started = perf_counter()
        prepared = prepared or self.prepare(clean_question, search_terms=trusted_search_terms)
        if prepared.question != clean_question or prepared.validation_input != validation_input:
            raise ValueError("Saved retrieval does not match this question/search_terms")
        evidence = prepared.retrieved_candidates
        retrieval_seconds = prepared.retrieval_seconds
        context_sources = prepared.context_sources

        if not context_sources:
            total_seconds = perf_counter() - total_started
            return QAResult(
                question=clean_question,
                answer=INSUFFICIENT_CONTEXT_ANSWER,
                cited_sources=[],
                retrieved_candidates=evidence,
                model=self.llm_client.model,
                top_k=self.config.top_k,
                candidate_k=self.config.candidate_k,
                max_context_chars=self.config.max_context_chars,
                context_characters=0,
                temperature=self.config.temperature,
                insufficient_context=True,
                partial_context=False,
                retrieval_trace=prepared.retrieval_trace,
                timings=QATimings(
                    retrieval_seconds=retrieval_seconds,
                    llm_seconds=0.0,
                    total_seconds=total_seconds,
                ),
            )

        user_prompt = build_user_prompt(clean_question, context_sources)
        completions = []
        llm_seconds = 0.0
        current_prompt = user_prompt
        for attempt in range(2):
            llm_started = perf_counter()
            try:
                completion = generate(
                    self.llm_client, stage="answer", operation=operation_id,
                    attempt=attempt, system_prompt=SYSTEM_PROMPT,
                    user_prompt=current_prompt, temperature=self.config.temperature,
                    max_tokens=self.config.max_output_tokens,
                )
                llm_seconds += perf_counter() - llm_started
                completions.append(completion)
                if not completion.content.strip():
                    raise EmptyResponseError("LLM returned an empty answer")
                cited_sources, insufficient, partial = _validate_completion(
                    completion.content,
                    clean_question,
                    context_sources,
                )
                assessment = parse_evidence(completion.content, clean_question, context_sources)
                if assessment is None:
                    assessment = legacy_assessment(cited_sources, insufficient=insufficient)
                partial = partial or assessment.conclusion == "requires_clarification"
                break
            except ContextValidationError:
                # A second generation cannot supply a missing normative value.
                # Reject the ungrounded answer and preserve its token usage.
                raise
            except (CitationValidationError, EmptyResponseError) as exc:
                if attempt == 1:
                    if isinstance(exc, EmptyResponseError):
                        raise CitationValidationError("LLM twice returned an empty answer") from exc
                    raise
                current_prompt = (
                    user_prompt
                    + "\n\nКРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ ОТВЕТА:\n"
                    + str(exc)
                    + "\nВерни JSON по схеме из system prompt. Исправь идентификаторы "
                    "источников, дословные выдержки и ссылки comparisons на их основания. "
                    "Не подставляй проектные значения вместо нормативных. "
                    "Для неизвестных условий используй unknowns и requires_clarification. "
                    "Не заменяй JSON свободным текстом."
                )

        answer_text = render_evidence(assessment, cited_sources)
        if partial and PARTIAL_CONTEXT_MARKER.casefold() not in answer_text.casefold():
            answer_text = answer_text.rstrip() + "\n\n" + PARTIAL_CONTEXT_MARKER

        return QAResult(
            question=clean_question,
            answer=answer_text,
            cited_sources=cited_sources,
            retrieved_candidates=evidence,
            model=completion.model,
            top_k=self.config.top_k,
            candidate_k=self.config.candidate_k,
            max_context_chars=self.config.max_context_chars,
            context_characters=len(user_prompt),
            temperature=self.config.temperature,
            insufficient_context=insufficient,
            partial_context=partial,
            timings=QATimings(
                retrieval_seconds=retrieval_seconds,
                llm_seconds=llm_seconds,
                total_seconds=perf_counter() - total_started,
            ),
            prompt_tokens=_sum_tokens(completions, "prompt_tokens"),
            completion_tokens=_sum_tokens(completions, "completion_tokens"),
            total_tokens=_sum_tokens(completions, "total_tokens"),
            validation=assessment.to_dict(),
            retrieval_trace=prepared.retrieval_trace,
        )

    def prepare(
        self, question: str, *, search_terms: Sequence[str] = (),
    ) -> PreparedQuestion:
        """Run unchanged retrieval once; persist before making an API call."""
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("Question cannot be empty")
        terms = [term.strip() for term in search_terms if isinstance(term, str) and term.strip()]
        started = perf_counter()
        evidence = self._retrieve_candidates(clean_question, terms)
        selection_started=perf_counter()
        context_sources = self._fit_context(evidence, clean_question, terms)
        self._retrieval_trace['context_selection_seconds']=perf_counter()-selection_started
        self._retrieval_trace['final_source_ids'] = [s.source_id for s in context_sources]
        self._retrieval_trace['context_characters'] = len(build_user_prompt(clean_question,context_sources))
        return PreparedQuestion(
            clean_question, " ".join([clean_question, *terms]), evidence,
            context_sources, perf_counter() - started, self._retrieval_trace,
        )

    def _retrieve_candidates(
        self,
        question: str,
        search_terms: Sequence[str],
    ) -> list[SourceEvidence]:
        plan = prepare_queries(question,search_terms)
        terms = list(plan.terms)
        queries = _compose_retrieval_queries(question, terms)
        unique: dict[tuple[object, ...], SearchResult] = {}
        origins: dict[tuple,list[dict]] = {}
        pool_depth = max(self.config.candidate_k, INTERNAL_POOL)
        self._retrieval_trace = {'version':VERSION,'selection_version':SELECTION_VERSION,
            'queries':queries,'clean_terms':terms,'records':[],'related_limit':2,
            'initial_depth':self.config.candidate_k,'internal_pool_depth':pool_depth}
        search_started = perf_counter()
        for query in queries:
            # A specific object phrase may be less semantically similar than a
            # broad regulatory formulation and therefore sit below the first
            # ten hits. Keep the configured candidate depth for every query;
            # the union is deduplicated and reduced by the lightweight
            # reranker before anything is sent to the LLM.
            search = getattr(self.retriever,'retrieve_candidates',self.retriever.retrieve)
            for position,result in enumerate(search(query,self.config.candidate_k),1):
                key = _result_key(result)
                origins.setdefault(key,[]).append({'query':query,'method':result.metadata.get('_retrieval_method','semantic'),
                    'initial_position':result.metadata.get('_initial_rank',position),'semantic_score':result.score,
                    'distance':result.distance,'lexical_score':result.metadata.get('_lexical_score')})
                existing = unique.get(key)
                if existing is None or result.score > existing.score:
                    unique[key] = SearchResult(result.distance,result.text,{**result.metadata,'_retrieval_query':query})
        self._retrieval_trace['search_seconds'] = perf_counter()-search_started
        for key,result in list(unique.items()):
            unique[key] = SearchResult(result.distance,result.text,{**result.metadata,'_provenance':origins[key]})
        rank_started=perf_counter()
        ranked = rerank_candidates(
            question,
            terms,
            list(unique.values()),
            top_k=max(1,len(unique)),
        )
        self._retrieval_trace['reranking_seconds']=perf_counter()-rank_started
        kept=ranked[:pool_depth]
        for rank,item in enumerate(ranked,1):
            self._retrieval_trace['records'].append({'chunk_id':chunk_key(item.result),'rerank_position':rank,
                'semantic_score':item.result.score,'rerank_score':item.rerank_score,'signals':item.signals,
                'hits':origins[_result_key(item.result)],'source_id':f'SOURCE_{rank}' if rank<=len(kept) else None,
                'decision':'candidate' if rank<=len(kept) else 'candidate_limit'})
        # Two exact existing chunks maximum. No recursive graph walk, no new
        # retrieval request, no mixing adjacent pages across documents.
        related_started=perf_counter()
        related = getattr(self.retriever,'related',None)
        if related is not None:
            extras=[]
            keys={_result_key(item.result) for item in kept}
            for item in kept[:self.config.top_k]:
                if item.result.score<self.config.min_relevance_score or chunk_kind(item.result.text) in {'contents','publication','references','title'}: continue
                for result in related(item.result,limit=2-len(extras)):
                    if _result_key(result) in keys: continue
                    keys.add(_result_key(result))
                    result=SearchResult(result.distance,result.text,{**result.metadata,'_provenance':[{
                        'query':item.result.metadata.get('_retrieval_query'), 'method':'related',
                        'initial_position':None,'semantic_score':None if result.metadata.get('_score_inherited') else result.score,
                        'parent_chunk_id':chunk_key(item.result),'reference':result.metadata.get('_reference')}]})
                    extras.append(result)
                    if len(extras)>=2: break
                if len(extras)>=2: break
            for result in extras:
                score = rerank_candidates(question,terms,[result],top_k=1)[0]
                kept.append(score)
                self._retrieval_trace['records'].append({'chunk_id':chunk_key(result),'rerank_position':None,
                    'semantic_score':None if result.metadata.get('_score_inherited') else result.score,
                    'rerank_score':score.rerank_score,'signals':score.signals,'hits':[{'method':'related',
                    'query':result.metadata.get('_provenance',[{}])[0].get('query'),'initial_position':None,
                    'parent_chunk_id':result.metadata.get('_related_to'),'reference':result.metadata.get('_reference'),
                    'reference_kind':result.metadata.get('_reference_kind')}],
                    'source_id':f'SOURCE_{len(kept)}','decision':'candidate'})
        self._retrieval_trace['related_seconds']=perf_counter()-related_started
        return [
            SourceEvidence.from_search_result(
                index,
                item.result,
                rerank_score=item.rerank_score,
                matched_terms=list(item.matched_terms),
            )
            for index, item in enumerate(kept, start=1)
        ]
    def _fit_context(
        self,
        sources: Sequence[SourceEvidence],
        question: str,
        search_terms: Sequence[str] = (),
    ) -> list[SourceEvidence]:
        selected: list[SourceEvidence] = []
        requested=aspects(question,search_terms)
        # Existing ordinary candidates may themselves be referenced conditions;
        # do not lose their dependency just because retrieval deduplicated them.
        result_rows=[SearchResult(s.distance,s.text,{'source_file':s.chunk_id.split(':p')[0],
                     'page':s.page,'chunk_index':i}) for i,s in enumerate(sources)]
        local_rows={chunk_key(r):s for r,s in zip(result_rows,sources)}
        by_chunk={s.chunk_id:s for s in sources}
        links=[]
        hints={}
        for parent,row in zip(sources,result_rows):
            for child in related_chunks(row,result_rows,limit=6):
                target=local_rows[chunk_key(child)]
                linked=replace(target,provenance=[{'method':'related','reference':child.metadata['_reference']}])
                links.append((parent,target,mandatory_link(parent,linked,requested)))
                location=explicit_location(parent.text)
                if location:
                    hints[target.source_id]={'location':location,'parent_source_id':parent.source_id,
                                             'reference':child.metadata['_reference']}
        roles={s.source_id:classify(s,requested,location_hint=hints.get(s.source_id,{}).get('location')) for s in sources}
        covered: set[str] = set()
        duplicate_pairs=set()
        for index,first in enumerate(sources):
            for second in sources[index+1:]:
                if duplicate(SearchResult(first.distance,first.text,{}),SearchResult(second.distance,second.text,{})):
                    duplicate_pairs.add(frozenset((first.chunk_id,second.chunk_id)))
        records={r['source_id']:r for r in self._retrieval_trace.get('records',[]) if r['source_id']}
        for sid,role in roles.items():
            if sid in hints: role['scope_hint']=hints[sid]
            if sid in records: records[sid]['selection']=role
        mandatory={s.source_id for s in sources if s.related_to in by_chunk
                   and mandatory_link(by_chunk[s.related_to],s,requested)}
        dependencies={s.source_id:[c.source_id for c in sources if c.related_to==s.chunk_id
                                  and c.source_id in mandatory] for s in sources}
        for parent,child,required in links:
            if required and len(dependencies[parent.source_id])<2 and child.source_id not in dependencies[parent.source_id]:
                dependencies[parent.source_id].append(child.source_id)
        # If an otherwise unqualified clause was selected using a spatial
        # cross-reference, retain that reference as provenance in context.
        for sid,hint in hints.items():
            parent_id=hint['parent_source_id']
            if (any(a.location==hint['location'] for a in requested) and roles[sid]['direct_aspects']
                    and explicit_location(next(s.text for s in sources if s.source_id==sid)) != hint['location']
                    and parent_id not in dependencies[sid]): dependencies[sid].append(parent_id)
        self._retrieval_trace['candidate_links']=[{'parent':p.source_id,'child':c.source_id,'mandatory_signal':required} for p,c,required in links]
        pending=list(sources)
        while pending and len(selected)<self.config.top_k:
            eligible=[]
            selected_ids={s.source_id for s in selected}
            needed={sid for parent in selected for sid in dependencies[parent.source_id] if sid not in selected_ids}
            reserve_due=bool(needed) and self.config.top_k-len(selected)<=len(needed)
            for source in pending:
                reason = None
                linked_parent=next((old for old in selected if old.chunk_id==source.related_to),None)
                if source.related_to and linked_parent is None: reason='awaiting_parent'
                elif source.score<self.config.min_relevance_score and not linked_parent: reason='below_min_score'
                elif chunk_kind(source.text) in {'publication','contents','references','title'}: reason='service_fragment'
                elif any(frozenset((source.chunk_id,old.chunk_id)) in duplicate_pairs for old in selected): reason='duplicate'
                elif len(build_user_prompt(question,[*selected,source]))>self.config.max_context_chars: reason='context_budget'
                else:
                    required_ids=set(dependencies[source.source_id])-selected_ids-{source.source_id}
                    required_rows=[s for s in sources if s.source_id in required_ids]
                    if source.source_id not in needed:
                        if len(selected)+1+len(required_rows)>self.config.top_k:
                            reason='dependency_slots'
                        elif len(build_user_prompt(question,[*selected,source,*required_rows]))>self.config.max_context_chars:
                            reason='dependency_budget'
                if reason:
                    if source.source_id in records: records[source.source_id]['decision']=reason
                    continue
                role=roles[source.source_id]
                new=set(role['direct_aspects'])-covered
                # Required dependencies have reserved remaining slots. Other
                # linked explanations never displace independent requirements.
                priority=(4 if reserve_due and source.source_id in needed else
                          3 if new else 2 if source.source_id in needed else
                          1 if role['direct_aspects'] else 0)
                relevance=source.rerank_score or 0.
                weights={a.label:1/(1+i*.15) for i,a in enumerate(requested)}
                gain=sum(weights[a] for a in new)/sum(weights.values()) if weights else 0.
                quality=max(role['matches'].values(),default=0.)
                utility=relevance+.40*gain+.20*quality
                eligible.append(((priority,utility),source))
            if not eligible: break
            _,best=max(eligible,key=lambda row:row[0])
            selected.append(best); pending.remove(best)
            if best.source_id in records:
                records[best.source_id]['decision']='included_related' if best.related_to else 'included_relevance_coverage'
                if best.source_id in mandatory:
                    records[best.source_id]['selection']['role']='mandatory_condition'
                records[best.source_id]['required_by']=[s.source_id for s in selected
                    if best.source_id in dependencies[s.source_id]]
                if best.source_id in needed and not roles[best.source_id]['direct_aspects']:
                    records[best.source_id]['selection']['role']='mandatory_condition'
            covered.update(roles[best.source_id]['direct_aspects'])
        for source in pending:
            if source.source_id in records:
                if records[source.source_id]['decision']=='candidate': records[source.source_id]['decision']='context_top_k'
                elif records[source.source_id]['decision']=='awaiting_parent': records[source.source_id]['decision']='parent_not_selected'
        selected_ids={s.source_id for s in selected}
        missing_dependencies=[{'parent':s.source_id,'missing_source_ids':[sid for sid in dependencies[s.source_id] if sid not in selected_ids]}
                              for s in selected if any(sid not in selected_ids for sid in dependencies[s.source_id])]
        self._retrieval_trace['aspect_coverage']={a.label:float(a.label in covered) for a in requested}
        self._retrieval_trace['uncovered_aspects']=[a.label for a in requested if a.label not in covered]
        self._retrieval_trace['missing_dependencies']=missing_dependencies
        self._retrieval_trace['coverage_method']='co-occurring object/parameter and modal lexical signals; not legal or semantic verification'
        self._retrieval_trace['completeness_verified']=False
        return selected


def _compose_retrieval_queries(
    question: str,
    search_terms: Sequence[str],
) -> list[str]:
    """Prepare a bounded, cleaned object/parameter query plan."""

    return list(prepare_queries(question,search_terms).queries)


def _resolve_citations(
    answer: str,
    available_sources: Sequence[SourceEvidence],
) -> list[SourceEvidence]:
    source_map = {source.source_id: source for source in available_sources}
    ordered_ids = _citation_ids(answer)
    invalid_ids = [source_id for source_id in ordered_ids if source_id not in source_map]
    if invalid_ids:
        raise CitationValidationError(
            "LLM cited source IDs that were not provided: " + ", ".join(invalid_ids)
        )
    return [source_map[source_id] for source_id in ordered_ids if source_id in source_map]


def _result_key(result: SearchResult) -> tuple[object, ...]:
    metadata = result.metadata
    return (
        metadata.get("source_file"),
        metadata.get("page"),
        metadata.get("chunk_index"),
    )


def _validate_measured_values(
    answer: str,
    question: str,
    cited_sources: Sequence[SourceEvidence],
    context_sources: Sequence[SourceEvidence],
) -> None:
    # Bind each citation-bearing span to its own sources. Adjacent citations
    # form one group. This is syntactic provenance, not semantic entailment.
    matches = list(_SOURCE_CITATION.finditer(answer))
    if matches:
        _resolve_citations(answer, context_sources)
        source_map = {source.source_id: source for source in cited_sources}
        cursor = 0
        index = 0
        while index < len(matches):
            match = matches[index]
            end = match.end()
            ids = re.findall(r"SOURCE_\d+", match.group())
            while index + 1 < len(matches) and re.fullmatch(r"[\s,;]*", answer[end:matches[index + 1].start()]):
                index += 1
                end = matches[index].end()
                ids.extend(re.findall(r"SOURCE_\d+", matches[index].group()))
            local_sources = [source_map[sid] for sid in dict.fromkeys(ids) if sid in source_map]
            _validate_measured_values(answer[cursor:match.start()], question, local_sources, context_sources)
            cursor = end
            index += 1
        # An uncited tail may repeat a project value, but cannot borrow a
        # normative value from an earlier citation somewhere in the answer.
        _validate_measured_values(answer[cursor:], question, [], context_sources)
        return
    without_citations = _SOURCE_CITATION.sub("", answer)
    claimed = _measurements(without_citations)
    if not claimed:
        return
    # Normalize each evidence text independently: joining unrelated chunks may
    # accidentally combine a trailing number with the next chunk's unit.
    allowed = _measurements(question)
    # A project value is not evidence for a normative minimum. This is a
    # conservative lexical rejection only, never a semantic acceptance rule.
    for match in _MEASURED_VALUE.finditer(answer):
        prefix = answer[max(0, match.start() - 70):match.start()]
        if re.search(r"(?:миним\w*|максим\w*|требуем\w*|норматив\w*|допустим\w*|не менее|не более)\s*(?:составляет|равен|равна|:)?\s*$", prefix, re.I):
            allowed.difference_update(_measurements(match.group()))
    for source in cited_sources:
        allowed.update(_measurements(source.text))
    invented = sorted(claimed - allowed)
    if invented:
        guidance = []
        missing_context = False
        for measurement in invented:
            matching_ids = [
                source.source_id
                for source in context_sources
                if measurement in _measurements(source.text)
            ]
            formatted = f"{measurement[0]}={measurement[1]}"
            if matching_ids:
                guidance.append(
                    formatted + " подтверждается " + ", ".join(matching_ids)
                )
            else:
                missing_context = True
                guidance.append(formatted + " отсутствует во всем контексте")
        error_class = ContextValidationError if missing_context else CitationValidationError
        error = error_class(
            "LLM answer contains measured values absent from the question and "
            "cited chunks. Исправь ссылки или удали значения: "
            + "; ".join(guidance)
        )
        error.diagnostics = _measurement_diagnostics(answer, question, cited_sources)
        raise error


def _validate_completion(
    answer: str,
    question: str,
    context_sources: Sequence[SourceEvidence],
) -> tuple[list[SourceEvidence], bool, bool]:
    assessment = parse_evidence(answer, question, context_sources)
    if assessment is not None:
        ids = list(dict.fromkeys(r["source_id"] for r in assessment.requirements))
        by_id = {s.source_id: s for s in context_sources}
        return ([by_id[sid] for sid in ids],
                assessment.conclusion == "insufficient_information",
                assessment.conclusion == "requires_clarification")
    cited_sources = _resolve_citations(answer, context_sources)
    folded = answer.casefold()
    has_insufficient_phrase = INSUFFICIENT_CONTEXT_ANSWER.casefold() in folded
    partial = _looks_partial(answer, question, cited_sources)
    insufficient = has_insufficient_phrase and not partial
    if not insufficient and not cited_sources:
        raise CitationValidationError(
            "LLM answer contains no citations to the provided sources"
        )
    _validate_measured_values(
        answer,
        question,
        cited_sources,
        context_sources,
    )
    # Free prose lacks a claim-level contract. Existing IDs and numbers alone
    # cannot make it a fully validated compliance decision.
    return cited_sources, insufficient, partial or (bool(cited_sources) and not insufficient)


def _looks_partial(
    answer: str,
    question: str,
    cited_sources: Sequence[SourceEvidence],
) -> bool:
    folded = answer.casefold()
    if PARTIAL_CONTEXT_MARKER.casefold() in folded:
        return True
    missing_detail = (
        (
            "не указаны" in folded
            or "отсутствуют" in folded
            or "не содерж" in folded
        )
        and any(
            marker in folded
            for marker in (
                "конкретные числовые",
                "дневн",
                "ночн",
                "детали",
                "дополнительн",
                "часть вопроса",
                "часть информации",
            )
        )
    )
    if missing_detail and cited_sources:
        return True
    insufficient_phrase = INSUFFICIENT_CONTEXT_ANSWER.casefold()
    if insufficient_phrase in folded and cited_sources:
        remainder = folded.replace(insufficient_phrase, "")
        remainder = _SOURCE_CITATION.sub("", remainder)
        return len(remainder.strip()) >= 80
    question_folded = question.casefold()
    if "дневн" in question_folded and "ночн" in question_folded:
        noise_values = {
            value
            for source in cited_sources
            for kind, value in _measurements(source.text)
            if kind == "noise_db"
        }
        if len(noise_values) < 2:
            return True
    return False


def _measurements(text: str) -> set[tuple[str, Decimal]]:
    """Canonical dimension/value pairs, with exact Decimal unit conversion."""
    return {(kind, value) for kind, value, _ in _measurement_matches(text)}


def _measurement_matches(text: str) -> list[tuple[str, Decimal, str]]:
    result = []
    for match in _MEASURED_VALUE.finditer(text):
        value = Decimal(match.group("value").replace(",", "."))
        unit = match.group("unit").casefold()
        if unit in {"м²", "м2"} or unit.startswith(("кв.", "квадратн")):
            kind = "area_m2"
        elif unit == "мм" or unit.startswith("миллиметр"):
            kind, value = "length_m", value / Decimal(1000)
        elif unit == "см" or unit.startswith("сантиметр"):
            kind, value = "length_m", value / Decimal(100)
        elif unit == "м" or unit.startswith("метр"):
            kind = "length_m"
        elif unit.startswith("дб"):
            kind = "noise_db"
        elif unit == "%" or unit.startswith("процент"):
            kind = "percent"
        else:
            kind = "people"
        result.append((kind, value, match.group(0)))
    result.extend(_width_table_matches(text))
    return result


def _width_table_matches(text: str) -> list[tuple[str, Decimal, str]]:
    """Read width cells only in explicit width(m)/slope tables flattened by PDF.

    Require a table heading, a dimensioned width header, an incline column,
    and a numeric width immediately followed by a slope ratio. Other bare
    numbers, document IDs, areas and the ratio components are not evidence.
    """
    result = []
    for table in re.finditer(r"\bТаблица\s+\d+(?:\.\d+)*", text, re.IGNORECASE):
        block = text[table.end():]
        stop = re.search(r"\bТаблица\s+\d|П\s*р\s*и\s*м\s*е\s*ч\s*а\s*н\s*и\s*я", block, re.IGNORECASE)
        block = block[:stop.start()] if stop else block
        header = re.search(r"ширина\s*,\s*м(?![\w²])\s+уклон\b", block, re.IGNORECASE)
        if not header:
            continue
        body = block[header.end():]
        # A cell may defer to another code; remove these references, preserving
        # offsets for debug evidence instead of mistaking an СП ID for a width.
        cleaned = re.sub(r"По\s+СП\s+\d+(?:\.\d+)*", lambda m: " " * len(m[0]), body, flags=re.IGNORECASE)
        for row in re.finditer(r"(?<![\w.,:])(?P<width>\d+(?:[.,]\d+)?)\s+"
                               r"\d+(?:[.,]\d+)?\s*:\s*\d+(?:[.,]\d+)?(?![\w.,])", cleaned):
            value = Decimal(row["width"].replace(",", "."))
            result.append(("length_m", value, "ширина, м: " + body[row.start():row.end()]))
    return result


def _measurement_diagnostics(answer: str, question: str,
                             cited_sources: Sequence[SourceEvidence]) -> list[dict]:
    """Local debug evidence for each claim; never treats uncited chunks as allowed."""
    units = {"length_m": "m", "area_m2": "m²", "noise_db": "dB", "percent": "%", "people": "people"}
    question_values = _measurement_matches(question)
    source_values = [(s.source_id, match) for s in cited_sources for match in _measurement_matches(s.text)]
    diagnostics = []
    for kind, value in sorted(_measurements(_SOURCE_CITATION.sub("", answer))):
        q_matches = [f"{raw} -> {normalized} {units[kind]}"
                     for dimension, normalized, raw in question_values if (dimension, normalized) == (kind, value)]
        s_matches = [{"source_id": source_id, "text": raw, "canonical": f"{normalized} {units[kind]}"}
                     for source_id, (dimension, normalized, raw) in source_values
                     if (dimension, normalized) == (kind, value)]
        diagnostics.append({"value": str(value), "dimension": kind,
                            "canonical": f"{value} {units[kind]}",
                            "question_matches": q_matches, "cited_source_matches": s_matches,
                            "rejected": not (q_matches or s_matches)})
    return diagnostics


def _sum_tokens(completions: Sequence[object], name: str) -> int | None:
    values = [getattr(completion, name, None) for completion in completions]
    known = [value for value in values if isinstance(value, int)]
    return sum(known) if known else None
