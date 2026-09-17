"""High-precision legal/official style checks that operate on edit spans.

The stage detects classes of errors rather than document-specific phrases:
adjacent repeated multi-word spans, malformed legal collocations and the
instrumental audit construction «в отношениях X» used instead of «в отношении X».
"""
from __future__ import annotations
import re
from decision_engine import EditCandidate

WORD_RE=re.compile(r"[А-Яа-яЁёA-Za-z]+(?:-[А-Яа-яЁёA-Za-z]+)*")
AUDIT_CUE=re.compile(r"\b(?:проверк(?:а|ой|е|и)|установлен[аоы]?|выявлен[аоы]?|обнаружен[аоы]?|нарушен[аоы]?|не\s+(?:разработан|утвержден|определен|установлен|представлен)[аоы]?)\b",re.IGNORECASE)
RELATION_ERROR=re.compile(r"\bв(?P<gap>\s+)отношениях(?P<object>\s+(?!между\b|с\b|по\b)[А-Яа-яЁё-]+)",re.IGNORECASE)
FIXED_RULES=(
 (re.compile(r"\bв\s+соответствие\s+с\b",re.IGNORECASE),"в соответствии с","устойчивая конструкция «в соответствии с»"),
 (re.compile(r"\bпо\s+истечению\b",re.IGNORECASE),"по истечении","нормативная временная конструкция «по истечении»"),
 (re.compile(r"\bпо\s+окончанию\b",re.IGNORECASE),"по окончании","нормативная временная конструкция «по окончании»"),
 (re.compile(r"\bпо\s+прибытию\b",re.IGNORECASE),"по прибытии","нормативная временная конструкция «по прибытии»"),
 (re.compile(r"\bв\s+течении(?=\s+(?:срока|дня|дней|месяца|месяцев|года|лет|периода)\b)",re.IGNORECASE),"в течение","производный предлог «в течение»"),
)

def _case_like(source,replacement):
 return replacement[:1].upper()+replacement[1:] if source[:1].isupper() else replacement

def _inside_quotes(text,pos):
 return text.rfind('«',0,pos+1)>text.rfind('»',0,pos+1)

class LegalStyleStage:
 def __init__(self):self.calls=0;self.found={'repetition':0,'government':0,'collocation':0}
 def _duplicates(self,text):
  words=list(WORD_RE.finditer(text));out=[];occupied=[]
  folded=[m.group().casefold() for m in words]
  for size in range(6,1,-1):
   i=0
   while i+2*size<=len(words):
    if folded[i:i+size]!=folded[i+size:i+2*size] or not any(len(x)>=4 for x in folded[i:i+size]):i+=1;continue
    first,second=words[i+size-1],words[i+size]
    if not re.fullmatch(r"\s+",text[first.end():second.start()]) or _inside_quotes(text,second.start()):i+=1;continue
    second_last=words[i+2*size-1];start=first.end();end=second_last.end()
    while end<len(text) and text[end] in ' \t\n':end+=1
    if any(not(end<=a or start>=b) for a,b in occupied):i+=1;continue
    before=text[start:end];newlines=before.count('\n')
    after='\n'*newlines if newlines else (' ' if end<len(text) and text[end] not in ',.;:!?)»' else '')
    out.append(EditCandidate(before,after,1.0,'rule-legal-repetition','удалён соседний повтор фразы',start=start,sources=('rule-legal-repetition',)))
    occupied.append((start,end));self.found['repetition']+=1;i+=2*size
  return out
 def _government(self,text):
  out=[]
  for m in RELATION_ERROR.finditer(text):
   left=max(0,text.rfind('.',0,m.start())+1);right=text.find('.',m.end());right=len(text) if right<0 else right+1
   if not AUDIT_CUE.search(text[left:right]):continue
   source='отношениях';start=m.start()+m.group(0).lower().find(source);after=_case_like(text[start:start+len(source)],'отношении')
   out.append(EditCandidate(text[start:start+len(source)],after,1.0,'rule-legal-government','в значении «касательно объекта» употребляется «в отношении»',start=start,sources=('rule-legal-government',)))
   self.found['government']+=1
  return out
 def _fixed(self,text):
  out=[]
  for pattern,replacement,reason in FIXED_RULES:
   for m in pattern.finditer(text):
    out.append(EditCandidate(m.group(),_case_like(m.group(),replacement),1.0,'rule-legal-collocation',reason,start=m.start(),sources=('rule-legal-collocation',)));self.found['collocation']+=1
  return out
 def candidates(self,text):
  self.calls+=1;return self._duplicates(text)+self._government(text)+self._fixed(text)
 def metrics(self):return {'calls':self.calls,**self.found}
