"""Быстрый нормативный поиск: лексика без Ollama, максимум один semantic-вызов."""
from __future__ import annotations
import re,sqlite3
from collections import defaultdict
_ABBR=re.compile(r"\b[А-ЯЁ]{2,12}\b");_WORD=re.compile(r"[А-Яа-яЁёA-Za-z0-9-]{3,}")
_STOP={'который','которые','находящихся','имеется','имеются','сотрудников','согласно','порядок','после','также','этого','данного','российской','федерации'}
_MARKERS=('должен','следует','не допускается','применяется','указывается','оформляется','именуется','сокращенное наименование','в случае','при наличии','для ')
def plan_queries(text):
 q=_ABBR.findall(text)
 for m in re.finditer(r"\b[А-ЯЁ]{2,12}\s+([^\n.]{3,160})",text):q.append(' '.join(m.group(1).split()))
 words=[w for w in _WORD.findall(text) if w.casefold() not in _STOP]
 if words:q.append(' '.join(words[:12]))
 out=[]
 for x in q:
  x=x.strip(' ,;:')
  if len(x)>1 and x.casefold() not in {y.casefold() for y in out}:out.append(x)
 return out[:5]
def _row(r,kind,score):return {'doc_id':r['doc_id'],'chunk_id':r['chunk_id'],'text':r['text'],'path':r['path'],'version':r['version'],'match_type':kind,'evidence_score':score}
def retrieve_evidence(store,embedder,text,top_k=3):
 queries=plan_queries(text);scores=defaultdict(float);items={}
 rows=list(store.db.execute("SELECT c.doc_id,c.chunk_id,c.text,d.path,d.version FROM chunks c JOIN documents d USING(doc_id) WHERE d.status='active'"))
 for qi,q in enumerate(queries):
  folded=q.casefold()
  for r in rows:
   if folded in r['text'].casefold():
    key=(r['doc_id'],r['chunk_id']);scores[key]+=3/(qi+1);items[key]=_row(r,'exact',scores[key])
  terms=list(dict.fromkeys(_WORD.findall(q.casefold())))[:10]
  if terms:
   try:
    fts=' AND '.join('"'+x.replace('"','')+'"' for x in terms)
    sql="SELECT c.doc_id,c.chunk_id,c.text,d.path,d.version FROM chunks_fts f JOIN chunks c ON c.id=f.rowid JOIN documents d ON d.doc_id=f.doc_id WHERE chunks_fts MATCH ? AND d.status='active' LIMIT 12"
    for rank,r in enumerate(store.db.execute(sql,(fts,)),1):
     key=(r['doc_id'],r['chunk_id']);scores[key]+=2/(qi+rank+1);items[key]=_row(r,'fts',scores[key])
   except sqlite3.OperationalError:pass
 # Ollama вызывается не более одного раза и только если лексики недостаточно.
 if len(items)<top_k:
  semantic_query=' '.join(_WORD.findall(text)[:20]) or text
  for rank,h in enumerate(store.search(semantic_query,top_k=top_k,embedder=embedder),1):
   key=(h['doc_id'],h['chunk_id']);scores[key]+=1/(rank+1);items.setdefault(key,h)
 ranked=sorted(items,key=lambda k:scores[k],reverse=True)[:top_k];result=[];terms=_ABBR.findall(text)+_WORD.findall(text)
 for key in ranked:
  h=dict(items[key]);src=h['text'];positions=[src.casefold().find(t.casefold()) for t in terms if len(t)>2 and src.casefold().find(t.casefold())>=0];p=min(positions) if positions else 0;start=max(0,p-180);h['excerpt']=src[start:start+650].strip();h['evidence_score']=round(scores[key],4);h['normative']=any(m in h['excerpt'].casefold() for m in _MARKERS) or '(' in h['excerpt'];result.append(h)
 return result
def render_evidence(hits):
 if not hits:return ''
 lines=['НОРМАТИВНЫЕ ДОКАЗАТЕЛЬСТВА. Проверяй область применения; не применяй справочный пример как обязательную норму.']
 for h in reversed(hits):lines.append(f"— ИСТОЧНИК {h['doc_id']}#{h['chunk_id']}: {h['excerpt']}")
 return '\n'.join(lines)
