"""Transparent lexical signals, not a legal applicability checker."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Sequence
from rag.models import SearchResult
from rag.query import prepare_queries, tokens, stem

VERSION = 'p1-3'
WEIGHTS = {'semantic': .35,'object': .30,'parameter': .20,'normative': .10,'scope': .05}
_MODAL = re.compile(r'долж\w*|следует|допуска\w*|не менее|не более|запрещ\w*|предусматри\w*',re.I)

def chunk_kind(text: str) -> str:
    folded, normative = text.casefold(),bool(_MODAL.search(text))
    if re.search(r'ключевые слова|редактор|корректор|подписано в печать',folded) and not normative: return 'publication'
    if ('содержание' in folded or len(re.findall(r'\.{3,}\s*\d+',text))>=3) and not normative: return 'contents'
    if re.search(r'нормативные ссылки|библиограф',folded) and not normative: return 'references'
    if re.search(r'утвержден|министерство|дата введения|предисловие',folded) and not normative and len(tokens(text))<50: return 'title'
    if re.search(r'термины и определения|\b\d+\.\d+\s+[^\n:]{3,100}:',text,re.I) and not normative: return 'definition'
    if 'таблица' in folded: return 'table'
    return 'body'

def duplicate(first: SearchResult, second: SearchResult) -> bool:
    a,b = ' '.join(first.text.casefold().split()),' '.join(second.text.casefold().split())
    if a==b: return True
    sig = lambda s:(set(re.findall(r'\d+(?:[.,]\d+)*',s)),set(re.findall(r'\b(?:кроме|если|условии|исключени\w*|не)\b',s)))
    if sig(a)!=sig(b): return False
    ta,tb=tokens(a),tokens(b)
    return bool(ta|tb) and len(ta&tb)/len(ta|tb)>=.90

@dataclass(frozen=True,slots=True)
class RerankedResult:
    result: SearchResult
    rerank_score: float
    matched_terms: tuple[str,...]
    signals: dict = field(default_factory=dict)

def rerank_candidates(query: str,search_terms: Sequence[str],candidates: Sequence[SearchResult],*,top_k: int)->list[RerankedResult]:
    if top_k<1: raise ValueError('top_k must be positive')
    plan,ranked=prepare_queries(query,search_terms),[]
    aspect_sets={a:tokens(a) for a in plan.aspects}
    term_sets={t:tokens(t) for t in plan.terms}
    qualified=[terms for t,terms in term_sets.items() if len(terms)>=2 and terms-plan.building-plan.parameters
               and re.search(r'\b[а-яё]+(?:ая|яя|ый|ий|ой|ое|ее|ые|ие|ых|их|ного|ной)\b',t,re.I)]
    fraction=lambda needs,words:len(needs&words)/len(needs) if needs else 0.
    for candidate in candidates:
        text,kind=candidate.text,chunk_kind(candidate.text)
        passages=[text] if kind=='table' else re.split(r'\n\s*\n|(?<=[;!?])\s+',text)
        if kind!='table': passages=[p if len(p)<=450 else p[i:i+450] for p in passages for i in range(0,max(1,len(p)),300)]
        # Qualified technical object phrases must match jointly. Sharing only
        # generic words ('path', 'room') cannot replace a compound object.
        passage_sets=[tokens(p) for p in passages]
        whole=tokens(text)
        def object_coverage(words):
            basic=fraction(plan.objects,words)
            return (basic+max((fraction(t,words) for t in qualified),default=0.))/2 if qualified else basic
        obj,param=max(((object_coverage(words),fraction(plan.parameters,words)) for words in passage_sets),key=lambda pair:pair[0]+pair[1],default=(0.,0.))
        modal=1. if _MODAL.search(text) or kind=='table' else 0.
        scope=fraction(plan.building,whole)
        incompatible=.15 if 'многоквартирн' in plan.building and re.search(r'силос|тоннел|палатн|зерн|одноквартирн',text,re.I) and not re.search(r'многоквартирн',text,re.I) else 0.
        penalty=.8 if kind in {'publication','contents','references','title'} else .2 if kind=='definition' and not plan.definition_request else 0.
        signals={'semantic':candidate.score,'object':obj,'parameter':param,'normative':modal,'scope':scope,'scope_penalty':incompatible,'service_penalty':penalty,'kind':kind,'aspect_coverage':{a:fraction(needs,whole) for a,needs in aspect_sets.items()}}
        score=sum(WEIGHTS[k]*signals[k] for k in WEIGHTS)-incompatible-penalty
        ranked.append(RerankedResult(candidate,score,tuple(t for t,needs in term_sets.items() if fraction(needs,whole)>=.75),signals))
    ranked.sort(key=lambda item:(-item.rerank_score,-item.result.score,str(item.result.metadata.get('source_file','')),item.result.metadata.get('chunk_index',0)))
    return ranked[:top_k]

def _same_lexeme(first: str,second: str)->bool:
    return stem(first)==stem(second)
