"""Bounded Russian query preparation; terminology, not ready-made answers."""
from __future__ import annotations
import re
from dataclasses import dataclass

_WORDS = re.compile(r"[a-zа-яё]+", re.I)
_STOP = set("а без в во для до и из или как какие какой какая какова каковы ли на не по при с со что это то так же об о к у мы нас вы он она они его ее нашего наш нужно надо проверить уточнить нормативный нормативные требования требование соответствовать соответствует соответствуют допускается допустимо можно минимум минимальный максимальный около примерно предварительно метр метра метров сантиметров миллиметров м см мм вопрос вопросов еще ещё уже вот хороший последний насколько понимаю вообще архитектор предлагает сделать предусмотрена предусмотрены получается заложена заложены должно быть должен должна ли этого достаточно именно касается речь идет обсуждается системы система случае контекст совещания уточняющий проект проектный решение устройство".split())
_STOP.update({'я','мне','нам','тебе','вам','требуется','предъявляются','допускают','применение','обеспечение','условия','основной','основного','один','разных','общий'})
_SUFFIXES = tuple(sorted(("иями", "ями", "ами", "ого", "ему", "ому", "ими", "ыми", "иях", "ах", "ях", "ией", "ий", "ый", "ой", "ая", "яя", "ое", "ее", "ые", "ие", "ую", "юю", "ых", "их", "ов", "ев", "ей", "ия", "ию", "ии", "иям", "а", "я", "ы", "и", "у", "ю", "е", "о", "ь"), key=len, reverse=True))
_IRREGULAR = {"пути": "путь", "путей": "путь", "путям": "путь", "путях": "путь", "путь": "путь", "кухонь": "кухн", "кухни": "кухн", "кухня": "кухн"}
PARAMETERS = {"ширин", "высот", "уклон", "площад", "уровен", "уровн", "шум", "освещенност", "воздухообмен", "объединен", "переток", "размещен", "смежн", "соседств", "нагрузк", "прочност", "температур", "расстоян", "толщин", "глубин", "количеств"}
BUILDING = {"многоквартирн", "жил", "жилой", "дом", "здан", "строен", "сооружен", "маломобильн", "групп", "населен", "доступност"}
_CONVERSATIONAL = re.compile(r"^(?:(?:и|а|вот|хорошо|тогда)\s+)*(?:(?:еще|ещё|последний)\s+)?(?:вопрос|это уже хороший вопрос)\s*(?:по|о|об)?\s*", re.I)
_PREFIX = re.compile(r"\b(вне|внутри|между)\s+([а-яё]{5,})\b", re.I)
# These are recall expansions, not assertions of interchangeable objects.
_VARIANTS = {"эвакуационная дверь": "эвакуационный выход", "дверь эвакуационного выхода": "эвакуационный выход", "высота двери": "высота дверного проема в свету", "ширина двери": "ширина дверного проема в свету", "аварийное освещение": "эвакуационное освещение", "подъемная платформа": "подъемная платформа вертикального перемещения"}
_VARIANTS.update({'лифтовая шахта':'шахта лифта'})
_VARIANTS.update({'вентиляционный канал':'вентиляционный воздуховод'})

def stem(word: str) -> str:
    word = word.casefold().replace('ё', 'е')
    if word in _IRREGULAR: return _IRREGULAR[word]
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word)-len(suffix) >= 3:
            base=word[:-len(suffix)]
            return base[:-4] if base.endswith('ционн') else base
    return word

_STOP_STEMS = {stem(w) for w in _STOP}

def tokens(text: str) -> set[str]:
    text = _PREFIX.sub(r"\1\2", text)
    return {stem(w) for w in _WORDS.findall(text) if w.casefold() not in _STOP and stem(w) not in _STOP_STEMS}

def coverage(needles: set[str], text: str) -> float:
    return len(needles & tokens(text))/len(needles) if needles else 0.0

@dataclass(frozen=True, slots=True)
class QueryPlan:
    original: str
    terms: tuple[str, ...]
    queries: tuple[str, ...]
    aspects: tuple[str, ...]
    objects: frozenset[str]
    parameters: frozenset[str]
    building: frozenset[str]
    definition_request: bool

def prepare_queries(question: str, search_terms=(), *, max_queries: int = 8) -> QueryPlan:
    if not question.strip() or max_queries < 1: raise ValueError('Нужен непустой вопрос и положительный лимит запросов')
    terms, seen = [], set()
    for raw in search_terms:
        if not isinstance(raw, str): continue
        term = _CONVERSATIONAL.sub('',raw.strip()).strip(' .,?!:;')
        term = re.sub(r'^(?:(?:и|а|вот|хорошо|тогда|еще|ещё)\s+)+','',term,flags=re.I)
        term = re.sub(r'^(?:\w+\s+)?(?:предлагает|предлагаем|предусмотрено|предусмотрена|заложена)\s+(?:сделать\s+)?','',term,flags=re.I)
        if len(_WORDS.findall(term)) > 8:
            term = ' '.join(w for w in _WORDS.findall(term) if w.casefold() not in _STOP)[:200]
        content = tokens(term)
        if content and frozenset(content) not in seen:
            terms.append(term); seen.add(frozenset(content))
    qt = tokens(question)
    all_tokens = qt | set().union(*(tokens(t) for t in terms)) if terms else qt
    params, objects, building = all_tokens & PARAMETERS, all_tokens-PARAMETERS-BUILDING, all_tokens & BUILDING
    # A bare parameter like 'height' is not a useful standalone technical
    # object query when an object is known. Preserve it in aspect metadata.
    contextual_terms=[t for t in terms if not tokens(t)<=PARAMETERS or not objects]
    queries = [question.strip(),*contextual_terms]
    expansions=[]
    if len(terms) >= 2:
        objects_terms = sorted((t for t in terms if tokens(t)&objects),key=lambda t:(-len(tokens(t)&objects),len(t)))
        param_terms = [t for t in terms if tokens(t)&params]
        expansions = [f'{p} {o}' for o in objects_terms[:2] for p in param_terms[:2] if o!=p]
    for old, variant in _VARIANTS.items():
        if coverage(tokens(old),question)==1:
            queries.append(variant+(' многоквартирное жилое здание' if 'многоквартирн' in building else ''))
    queries += expansions
    clauses = re.split(r'[,;?]|\s+и\s+(?=(?:допуска|можно|каков|какие|треб|соответ|есть))',question,flags=re.I)
    aspects = list(dict.fromkeys([t for t in terms if tokens(t)-BUILDING]+[c.strip() for c in clauses if tokens(c)-BUILDING]))
    return QueryPlan(question.strip(),tuple(terms),tuple(dict.fromkeys(queries))[:max_queries],tuple(aspects[:12]),frozenset(objects),frozenset(params),frozenset(building),bool(re.search(r'что (?:такое|означает)|определение|термин',question,re.I)))
