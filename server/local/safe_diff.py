"""Извлечение понятных локальных правок из пары «исходник → результат модели»."""
from __future__ import annotations
import difflib,math,re
from decision_engine import EditCandidate

WORD_RE=re.compile(r"[А-Яа-яЁёA-Za-z]+")
TOKEN_RE=re.compile(r"[А-Яа-яЁёA-Za-z0-9]+|[^\sА-Яа-яЁёA-Za-z0-9]")
CONTEXT_RE=re.compile(r"[А-Яа-яЁёA-Za-z0-9]+(?:[.-][А-Яа-яЁёA-Za-z0-9]+)*")
MAX_FRAGMENT_CHARS=90

def _tokens(text):return [(m.group(0),m.start(),m.end()) for m in TOKEN_RE.finditer(text)]
def _global_rewrite(source,corrected):
 source_words=WORD_RE.findall(source);corrected_words=WORD_RE.findall(corrected)
 if source_words and abs(len(corrected_words)-len(source_words))>1:return True
 baseline=max(1,min(len(source),len(corrected)))
 if abs(len(corrected)-len(source))/baseline>.20:return True
 matcher=difflib.SequenceMatcher(None,source_words,corrected_words,autojunk=False);changed=0;replace_ops=0
 for tag,i1,i2,j1,j2 in matcher.get_opcodes():
  if tag=='equal':continue
  changed+=max(i2-i1,j2-j1);replace_ops+=tag=='replace'
  if i2-i1>2 or j2-j1>2:return True
 return changed>max(3,math.ceil(max(len(source_words),1)*.12)) or replace_ops>3

def _right_context(text,pos,limit=5):
 words=[m.group() for m in CONTEXT_RE.finditer(text,pos)]
 return ' '.join(words[:limit])
def _reason(source,start,before,after,category):
 if before.count(',')==after.count(',')+1 and before.replace(',','',1)==after:
  comma=start+before.find(',');ctx=_right_context(source,comma+1)
  return f'Удалена лишняя запятая перед оборотом «{ctx}».' if ctx else 'Удалена лишняя запятая.'
 if after.count(',')==before.count(',')+1 and after.replace(',','',1)==before:
  comma=after.find(',');ctx=' '.join(CONTEXT_RE.findall(after[comma+1:])[:5])
  return f'Поставлена необходимая запятая перед «{ctx}».' if ctx else 'Поставлена необходимая запятая между частями предложения.'
 if re.fullmatch(r'[А-Яа-яЁё-]+',before) and re.fullmatch(r'[А-Яа-яЁё-]+',after):
  if category.startswith(('sage','spell')):return 'Исправлена орфографическая форма слова.'
  return 'Исправлена грамматическая форма слова по контексту.'
 if all(not ch.isalnum() for ch in before+after):return 'Исправлен знак препинания по структуре предложения.'
 if category.startswith('russian-gec-reasoning'):return 'Уточнена грамматическая конструкция по контексту предложения.'
 if category.startswith('draft'):return 'Исправлена грамматическая конструкция после независимой проверки.'
 return 'Внесена точечная языковая правка по контексту.'

def diff_candidates(source,corrected,category,confidence=.70,offset=0):
 if not source or not corrected or source==corrected:return []
 if source.count('\n')!=corrected.count('\n') or len(source.split('\n\n'))!=len(corrected.split('\n\n')):return []
 if _global_rewrite(source,corrected):return []
 src=_tokens(source);dst=_tokens(corrected)
 if not src or not dst:return []
 opcodes=difflib.SequenceMatcher(None,[t[0] for t in src],[t[0] for t in dst],autojunk=False).get_opcodes();out=[]
 for _,(tag,i1,i2,j1,j2) in enumerate(opcodes):
  if tag=='equal':continue
  if i1==i2 or j1==j2:
   i1=max(0,i1-1);i2=min(len(src),i2+1);j1=max(0,j1-1);j2=min(len(dst),j2+1)
  if i1>=i2 or j1>=j2:return []
  start,end=src[i1][1],src[i2-1][2];before=source[start:end];after=corrected[dst[j1][1]:dst[j2-1][2]]
  if '\n' in before or '\n' in after or not before or not after:return []
  if len(before)>MAX_FRAGMENT_CHARS or len(after)>MAX_FRAGMENT_CHARS:return []
  if before==after:continue
  out.append(EditCandidate(before,after,confidence,category,_reason(source,start,before,after,category),start=offset+start))
 return out
