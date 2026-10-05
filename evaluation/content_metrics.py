"""Offline, timestamp-free matching. No learned models or API calls."""
from collections import Counter
from functools import lru_cache
import math
from evaluation.metrics import normalize_text


def text_similarity(a, b):
    """Cosine of character trigram counts after the existing normalization."""
    def features(text):
        text = ' ' + normalize_text(text) + ' '
        return Counter(text[i:i+3] for i in range(max(0, len(text)-2)))
    x, y = features(a), features(b)
    denominator = math.sqrt(sum(v*v for v in x.values()) * sum(v*v for v in y.values()))
    return sum(v*y[k] for k, v in x.items()) / denominator if denominator else 0.0


def issue_similarity(reference, predicted):
    text = predicted.get('normalized_question', predicted.get('question', ''))
    words = normalize_text(text).split()
    # AND across groups, OR across prefix variants. No answer/source/timestamp input.
    for group in reference.get('match_concepts', []):
        if not any(word.startswith(prefix) for word in words for prefix in group):
            return 0.0
    return text_similarity(reference['question'], text)


def exact_word_alignment(reference, hypothesis):
    """Full Levenshtein DP; ties diagonal, deletion, insertion; return exact pairs."""
    n, m = len(reference), len(hypothesis)
    trace = [bytearray(m+1) for _ in range(n+1)]
    previous = list(range(m+1))
    for j in range(1, m+1): trace[0][j] = 2
    for i in range(1, n+1):
        trace[i][0] = 1
        current = [i]
        for j in range(1, m+1):
            options = (previous[j-1] + (reference[i-1] != hypothesis[j-1]),
                       previous[j]+1, current[-1]+1)
            step = min(range(3), key=lambda k: options[k])
            trace[i][j] = step
            current.append(options[step])
        previous = current
    pairs=[]; i=n; j=m
    while i or j:
        step=trace[i][j]
        if step == 0:
            if reference[i-1] == hypothesis[j-1]: pairs.append((i-1,j-1))
            i-=1; j-=1
        elif step == 1: i-=1
        else: j-=1
    return list(reversed(pairs))


def speaker_attribution_metrics(reference, hypothesis, *, min_word_coverage=.5):
    method = ('Global word Levenshtein alignment, exact-word evidence only; '
              'ties diagonal/deletion/insertion; per-turn majority speaker; '
              'one-to-one anonymous label mapping maximizes correct matched turns; '
              'mapping ties use sorted labels. No timestamps scored.')
    if reference is None:
        return {'speaker_attribution_accuracy':None,'reason':'reference utterances not provided', 'method':method}
    rw=[]; owners=[]; hw=[]; speakers=[]
    for i, turn in enumerate(reference):
        words=normalize_text(turn['text']).split(); rw.extend(words); owners.extend([i]*len(words))
    for turn in hypothesis:
        words=normalize_text(turn['text']).split(); hw.extend(words); speakers.extend([turn['speaker']]*len(words))
    evidence=[Counter() for _ in reference]
    for a,b in exact_word_alignment(rw,hw): evidence[owners[a]][speakers[b]]+=1
    rows=[]
    for i, turn in enumerate(reference):
        total=len(normalize_text(turn['text']).split()); counts=evidence[i]
        coverage=sum(counts.values())/total if total else 0
        winners=sorted(k for k,v in counts.items() if v==max(counts.values(),default=0))
        label=winners[0] if len(winners)==1 and counts[winners[0]]>sum(counts.values())/2 else None
        rows.append({'reference_id':turn.get('id',str(i+1)), 'reference_speaker':turn['speaker'],
                     'hypothesis_speaker':label,'word_coverage':coverage,
                     'matched':bool(total and coverage>=min_word_coverage),
                     'speaker_votes':dict(sorted(counts.items()))})
    refs=sorted({t['speaker'] for t in reference})
    labels=sorted({r['hypothesis_speaker'] for r in rows if r['matched'] and r['hypothesis_speaker'] is not None
                   and r['hypothesis_speaker'].lower() not in ('unknown','unknown_speaker','speaker_unknown','неизвестно','неизвестный')})
    # Small number of meeting speakers: exact assignment with memoized bit masks.
    weights=[[sum(r['matched'] and r['hypothesis_speaker']==label and r['reference_speaker']==ref for r in rows)
              for label in labels] for ref in refs]
    @lru_cache(None)
    def assign(i, used):
        if i==len(refs): return (0,())
        choices=[]
        for j in range(len(labels)):
            if not used & (1<<j):
                score,path=assign(i+1,used|(1<<j)); choices.append((score+weights[i][j],(j,)+path))
        score,path=assign(i+1,used); choices.append((score,(len(labels),)+path))
        return min(choices,key=lambda v:(-v[0],v[1]))
    _, path=assign(0,0)
    mapping={labels[j]:refs[i] for i,j in enumerate(path) if j<len(labels)}
    for row in rows:
        row['mapped_speaker']=mapping.get(row['hypothesis_speaker'])
        row['correct']=row['matched'] and row['mapped_speaker']==row['reference_speaker']
    matched=sum(r['matched'] for r in rows); correct=sum(r['correct'] for r in rows)
    return {'speaker_attribution_accuracy':correct/matched if matched else None,
            'reference_utterances':len(reference),'matched_utterances':matched,
            'unmatched_utterances':len(reference)-matched,'correct_utterances':correct,
            'incorrect_utterances':matched-correct,'matching_coverage':matched/len(reference) if reference else None,
            'min_word_coverage':min_word_coverage,'speaker_mapping':mapping,
            'per_utterance':rows,'method':method,'reason':None if matched else 'no matched utterances'}
