"""Планирование нормативного поиска и компактная упаковка доказательств."""
from __future__ import annotations
import re
from collections import defaultdict

_ABBR=re.compile(r"\b[А-ЯЁ]{2,12}\b")
_WORD=re.compile(r"[А-Яа-яЁёA-Za-z0-9-]{3,}")
_STOP={'который','которые','находящихся','имеется','имеются','сотрудников','согласно','порядок','после','также','этого','данного','российской','федерации'}
_MARKERS=('должен','должны','следует','не допускается','применяется','указывается','оформляется','именуется','сокращенное наименование','в случае','при наличии','для ')

def plan_queries(text:str)->list[str]:
    queries=[]
    abbreviations=_ABBR.findall(text)
    queries.extend(abbreviations)
    # Фраза после сокращения важнее ошибочного сокращения: по ней находится канон.
    for match in re.finditer(r"\b[А-ЯЁ]{2,12}\s+((?:«[^\n.]{3,160}|[^\n.]{3,160}))",text):
        phrase=' '.join(match.group(1).split())
        if phrase:queries.append(phrase)
    significant=[w for w in _WORD.findall(text) if w.casefold() not in _STOP]
    if significant:queries.append(' '.join(significant[:12]))
    queries.append(' '.join(text.split()))
    out=[]
    for q in queries:
        q=q.strip(' ,;:')
        if len(q)>=2 and q.casefold() not in {x.casefold() for x in out}:out.append(q)
    return out[:6]

def _focus(text:str,terms:list[str],limit=650)->str:
    folded=text.casefold();positions=[folded.find(t.casefold()) for t in terms if len(t)>=3 and folded.find(t.casefold())>=0]
    center=min(positions) if positions else 0;start=max(0,center-limit//3);end=min(len(text),start+limit)
    return text[start:end].strip()

def retrieve_evidence(store,embedder,text:str,top_k=3):
    scores=defaultdict(float);items={};queries=plan_queries(text)
    for q_index,query in enumerate(queries):
        for rank,hit in enumerate(store.search(query,top_k=max(4,top_k),embedder=embedder),1):
            key=(hit['doc_id'],hit['chunk_id']);kind=hit.get('match_type','semantic')
            bonus={'exact':3.0,'fts':2.0,'semantic':1.0}.get(kind,1.0)
            scores[key]+=bonus/(rank+q_index+1);items[key]=hit
    ranked=sorted(items,key=lambda key:scores[key],reverse=True)[:top_k]
    terms=_ABBR.findall(text)+[w for w in _WORD.findall(text) if w.casefold() not in _STOP]
    result=[]
    for key in ranked:
        hit=dict(items[key]);hit['evidence_score']=round(scores[key],4);hit['excerpt']=_focus(hit['text'],terms)
        lower=hit['excerpt'].casefold();hit['normative']=any(marker in lower for marker in _MARKERS) or '(' in hit['excerpt']
        result.append(hit)
    return result

def render_evidence(hits:list[dict])->str:
    if not hits:return ''
    lines=['НОРМАТИВНЫЕ ДОКАЗАТЕЛЬСТВА. Сначала установи область применения каждого пункта. Не применяй правило, если условие или вид документа не совпадает. Канонические наименования и определения применяй точно.']
    # Лучшее доказательство последнее: локальный reasoner читает хвост контекста.
    for hit in reversed(hits):
        kind='нормативное' if hit.get('normative') else 'справочное'
        lines.append(f"— ИСТОЧНИК {hit['doc_id']}#{hit['chunk_id']} ({kind}): {hit['excerpt']}")
    return '\n'.join(lines)
