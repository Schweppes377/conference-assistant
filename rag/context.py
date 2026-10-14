"""Context diversity and explicit same-document links; no PDF re-ingestion."""
from __future__ import annotations
import re
from rag.models import SearchResult

def chunk_key(result: SearchResult) -> str:
    m=result.metadata
    return f"{m.get('source_file')}:p{m.get('page')}:c{m.get('chunk_index')}"

def referenced_targets(text: str) -> list[tuple[str,str]]:
    targets=[]
    for match in re.finditer(r'(?:(?:по|согласно)\s+|пункт(?:у|а|е|ом)?\s+)(\d+(?:\.\d+){1,3})(?:\s*[–—-]\s*(\d+(?:\.\d+){1,3}))?',text,re.I):
        # Qualified foreign references are not presumed to refer to this PDF.
        before=text[max(0,match.start()-50):match.start()]
        if re.search(r'(?:СП|ГОСТ|СанПиН)\s+[\d.]+[^.;\n]*$',before,re.I): continue
        start,end=match[1],match[2]
        targets.append(('clause',start))
        if end:
            a,b=start.split('.'),end.split('.')
            if a[:-1]==b[:-1] and 0<int(b[-1])-int(a[-1])<=3:
                targets += [('clause','.'.join(a[:-1]+[str(n)])) for n in range(int(a[-1])+1,int(b[-1])+1)]
    targets += [('table',m[1]) for m in re.finditer(r'таблиц(?:е|ы|у|ей)\s+(\d+(?:\.\d+)*)',text,re.I)]
    return list(dict.fromkeys(targets))[:6]

def related_chunks(parent: SearchResult, corpus: list[SearchResult], *, limit: int=2) -> list[SearchResult]:
    if limit<1: return []
    found=[]
    for kind,label in referenced_targets(parent.text):
        pattern=(r'\bТ\s*а\s*б\s*л\s*и\s*ц\s*а\s+'+re.escape(label)+r'(?![\d.])') if kind=='table' else r'(?<![\w.])'+re.escape(label)+r'(?:\.(?!\d))?\s+[А-ЯЁ]'
        for row in corpus:
            if row.metadata.get('source_file')!=parent.metadata.get('source_file') or chunk_key(row)==chunk_key(parent): continue
            if re.search(pattern,row.text,re.I if kind=='table' else 0):
                if any(chunk_key(r)==chunk_key(row) for r in found): continue
                found.append(SearchResult(parent.distance,row.text,{**row.metadata,
                    '_retrieval_method':'related','_related_to':chunk_key(parent),
                    '_reference_kind':kind,'_reference':label,'_score_inherited':True}))
                if len(found)>=limit: return found
    return found
