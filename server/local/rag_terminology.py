"""Детерминированная проверка сокращённых наименований по активному RAG."""
from __future__ import annotations
import os,re,time
from pathlib import Path
from decision_engine import EditCandidate
from shared.rag_store import RagStore

_ABBR=re.compile(r"^[А-ЯЁ]{2,12}$")
_PARENS=re.compile(r"\(([^()\n]{5,240})\)")


def _distance(a:str,b:str)->int:
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1):cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=cur
    return prev[-1]


def _tail_pattern(tail:str)->str:
    return r"\s+".join(re.escape(x) for x in re.split(r"\s+",tail.strip()))

class RagTerminologyStage:
    def __init__(self,store_dir:str|None=None):
        self.enabled=os.getenv('RAG_ENABLED','true').lower() in {'1','true','yes','on'}
        self.store=RagStore(store_dir or os.getenv('RAG_STORE_DIR','/home/service/llama/RAG/state')) if self.enabled else None
        self._aliases=[];self._refreshed=0.0;self._calls=0;self._matches=0

    def _refresh(self):
        if not self.store or time.monotonic()-self._refreshed<30:return
        variants={}
        rows=self.store.db.execute("SELECT c.text,c.doc_id FROM chunks c JOIN documents d USING(doc_id) WHERE d.status='active'")
        for row in rows:
            for found in _PARENS.finditer(row['text']):
                content=' '.join(found.group(1).split());parts=content.split(None,1)
                if len(parts)!=2 or not _ABBR.fullmatch(parts[0]) or '«' not in parts[1]:continue
                tail=parts[1].strip();key=tail.casefold();variants.setdefault(key,[]).append((parts[0],tail,row['doc_id']))
        aliases=[]
        for records in variants.values():
            abbreviations={x[0] for x in records}
            if len(abbreviations)==1:
                abbr,tail,doc_id=records[0];aliases.append((abbr,tail,doc_id,re.compile(r"\b(?P<abbr>[А-ЯЁ]{2,12})\s+"+_tail_pattern(tail),re.IGNORECASE)))
        self._aliases=aliases;self._refreshed=time.monotonic()

    def candidates(self,text:str):
        if not self.enabled or not self.store:return []
        self._calls+=1;self._refresh();out=[];seen=set()
        for canonical,tail,doc_id,pattern in self._aliases:
            for match in pattern.finditer(text):
                actual=match.group('abbr')
                if actual==canonical or _distance(actual,canonical)>2:continue
                key=(match.start('abbr'),actual,canonical)
                if key in seen:continue
                seen.add(key);out.append(EditCandidate(actual,canonical,1.0,'rule-rag-terminology',f'сокращённое наименование по {Path(doc_id).name}',match.start('abbr'),('rule-rag-terminology',)))
        self._matches+=len(out);return out

    def metrics(self):return {'enabled':self.enabled,'aliases':len(self._aliases),'calls':self._calls,'matches':self._matches}
