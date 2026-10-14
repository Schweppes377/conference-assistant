"""Bounded context allocation. Lexical roles are NOT legal applicability proof."""
from __future__ import annotations

import re
from dataclasses import dataclass
from rag.query import prepare_queries, tokens as _tokens, PARAMETERS, BUILDING
from rag.reranker import chunk_kind

VERSION = 'p1.1-1'
INTERNAL_POOL = 40


def tokens(text: str) -> set[str]:
    aliases={'уровен':'уровн','лифтов':'лифт'}
    return {aliases.get(w,w) for w in _tokens(text)}


_PARAMETERS = (PARAMETERS - {'шум'}) | tokens('влажность')
_MODAL = re.compile(r'долж\w*|следует|необходимо|не менее|не более|допуска\w*|запрещ\w*|предусматри\w*', re.I)
_CONDITION = re.compile(r'при условии|за исключением|исключени\w*|кроме|если', re.I)
_PARAM = re.compile(r'\b(?:ширин\w*|высот\w*|уклон\w*|толщин\w*|глубин\w*|температур\w*|площад\w*|уровн\w*|количеств\w*)', re.I)
# Spatial qualifiers, not document/page/answer rules. A matching word alone
# never establishes applicability; conflicts remain explicit diagnostics.
_LOCATIONS = {'entrance': r'\bвход\w*|\bподъезд\w*|\bнаружн\w*|\bвнешн\w*',
              'internal': r'\bвнутри\b|\bвнутренн\w*|между этажами|на этажи выше|ниже этажа',
              'stage': r'\bсцен\w*|\bтрибун\w*'}


@dataclass(frozen=True)
class Aspect:
    label: str
    objects: frozenset[str]
    parameters: frozenset[str]
    location: str | None = None


def aspects(question: str, terms=()) -> list[Aspect]:
    """Bind standalone parameters to their nearby object, not whole question."""
    plan = prepare_queries(question, terms)
    compact = [t for t in plan.terms if len(tokens(t)) <= 6 and not re.search(
        r'\b(?:сейчас|заложен\w*|хотим|предлага\w*|получается|предварительно|этого|достаточно)\b',t,re.I)]
    rows: list[Aspect] = []
    seen = set()
    location = 'entrance' if re.search(_LOCATIONS['entrance'], question, re.I) else None

    def add(label, objects, parameters):
        if not objects and not parameters: return
        key = (frozenset(objects), frozenset(parameters))
        if key not in seen:
            rows.append(Aspect(label, key[0], key[1], location)); seen.add(key)

    for term in compact:
        words = tokens(term)
        params, obj = words & _PARAMETERS, words - _PARAMETERS - BUILDING
        if params and obj: add(term, obj, params)

    object_terms = [(t, tokens(t)-_PARAMETERS-BUILDING) for t in compact
                    if tokens(t)-_PARAMETERS-BUILDING and not tokens(t)&_PARAMETERS]
    for match in _PARAM.finditer(question):
        param = tokens(match.group()) & _PARAMETERS
        if not param: continue
        following = tokens(question[match.end():match.end()+85])
        choices = [(t, obj) for t, obj in object_terms if obj <= following]
        if choices:
            label, obj = min(choices, key=lambda pair: len(pair[1]))
            if not any(param <= a.parameters and obj <= a.objects for a in rows):
                add(match.group()+' '+label, obj, param)

    for term, obj in object_terms:
        if any(obj <= a.objects for a in rows): continue
        # Context words alone (day/night) cannot establish a requirement.
        if obj <= {'ден', 'день', 'ноч', 'ночн'}:
            noise = next((a for a in rows if 'шум' in a.parameters or 'шум' in a.objects), None)
            if noise: add(noise.label+' '+term, noise.objects | obj, noise.parameters)
            continue
        if len(obj) == 1 and plan.parameters and rows and obj <= {'двер', 'эвакуац', 'лестниц'}:
            continue
        add(term, obj, frozenset())

    # Entrance is a requested independent object even when detection only
    # emitted the generic keyword "accessibility".
    if location:
        add('доступность входа', tokens('вход'), tokens('доступный'))
    if not rows:
        add(question, plan.objects, plan.parameters)
    return rows[:12]


def passages(text: str) -> list[str]:
    """Keep clauses/lists together; do not split decimal numbers or PDF lines."""
    clean = ' '.join(text.split())
    # Sentence punctuation only; numeric clause identifiers remain intact.
    sentences = re.split(r'(?<=[.!?;])\s+(?=[А-ЯЁа-яё-]|\d+\.\d+)', clean)
    return [s for s in sentences if s] or [clean]


def classify(source, requested: list[Aspect], *, location_hint: str | None = None) -> dict:
    kind = chunk_kind(source.text)
    if kind in {'publication', 'contents', 'references', 'title'}:
        return {'role': 'service_or_irrelevant', 'direct_aspects': [], 'matches': {}, 'scope_conflicts': []}
    chunks = passages(source.text)
    chunk_location = location_hint
    if kind == 'table': chunks.append(' '.join(source.text.split()))
    direct, matches, conflicts = [], {}, []
    for aspect in requested:
        best = 0.0
        for passage in chunks:
            # A condition mentioning the query object is not itself the main
            # regulated object (e.g. seating permitted IF lighting is present).
            passage=re.split(r',?\s+при\s+(?:наличии|ширине|условии)|,\s*(?:если|когда|с учетом)|\bза исключением\b|\bкроме\b',passage,maxsplit=1,flags=re.I)[0]
            words = tokens(passage)
            obj = len(aspect.objects & words)/len(aspect.objects) if aspect.objects else 1.
            param = len(aspect.parameters & words)/len(aspect.parameters) if aspect.parameters else 1.
            scope_conflict = bool(aspect.location and
                any(re.search(pattern, passage, re.I) for name, pattern in _LOCATIONS.items() if name != aspect.location))
            if scope_conflict:
                conflicts.append(aspect.label); continue
            signal = obj * param
            # Requested dimension must modify the requested object nearby,
            # not another noun elsewhere in the same normative sentence.
            ordered=[tokens(w) for w in re.findall(r'[а-яёa-z]+',passage,re.I)]
            bound=kind=='table' or not aspect.objects or not aspect.parameters or any(
                aspect.objects <= set().union(*ordered[max(0,i-3):i+4])
                for i,word in enumerate(ordered) if word & aspect.parameters)
            located=not aspect.location or chunk_location==aspect.location or bool(re.search(_LOCATIONS[aspect.location],passage,re.I))
            # Specific phrase + its requested parameter must occur in a
            # normative sentence/table, not separately anywhere in a chunk.
            if obj >= .75 and param == 1 and bound and located and (_MODAL.search(passage) or kind == 'table'):
                direct.append(aspect.label)
                # Matched entrance/external scope outranks unqualified text.
                signal += .25 if aspect.location and located else 0.
            best = max(best, signal)
        matches[aspect.label] = best
    direct = list(dict.fromkeys(direct))
    role = ('direct_requirement' if direct else 'related_explanation' if source.related_to
            else 'general_context' if max(matches.values(), default=0.) > 0 else 'service_or_irrelevant')
    return {'role': role, 'direct_aspects': direct, 'matches': matches,
            'scope_conflicts': list(dict.fromkeys(conflicts)),
            'conditions_in_chunk': bool(_CONDITION.search(source.text)),
            'applicability_verified': False}


def explicit_location(text: str) -> str | None:
    if re.search(_LOCATIONS['entrance'],text,re.I) and not re.search(_LOCATIONS['internal']+'|'+_LOCATIONS['stage'],text,re.I):
        return 'entrance'
    return None


def mandatory_link(parent, child, requested: list[Aspect]) -> bool:
    """A condition or normative cross-reference in a relevant parent clause.

    Conservative signal: preserve dependency, never certify legal scope.
    Unrelated references elsewhere in the parent do not get reserved slots.
    """
    refs = [h.get('reference') for h in (child.provenance or []) if h.get('method') == 'related']
    for passage in passages(parent.text):
        if refs and not any(ref and re.search(r'(?<![\d.])'+re.escape(str(ref))+r'(?!\d|\.\d)', passage) for ref in refs):
            continue
        words = tokens(passage)
        relevant = any(a.objects <= words and (not a.parameters or a.parameters <= words) for a in requested)
        inherited = False
        if not relevant:
            # List item inherits its introductory normative clause only when
            # an actual colon connects them, not the whole neighbouring chunk.
            clean=' '.join(parent.text.split())
            position=clean.find(passage)
            prefix=clean[max(0,position-700):position] if position>=0 else ''
            if ':' in prefix:
                introduction=prefix.rsplit(':',1)[0]
                words=tokens(introduction)
                relevant=bool(_MODAL.search(introduction)) and any(
                    a.objects <= words and (not a.parameters or a.parameters <= words) for a in requested)
                inherited=relevant
        if relevant and (inherited or _CONDITION.search(passage) or _MODAL.search(passage)):
            return True
    return False
