"""Morphology-backed government and agreement frames for official Russian."""
from __future__ import annotations
import re
from decision_engine import EditCandidate
from morphology import get_morphology,preserve_capitalization,preserve_yo
WORD_RE=re.compile(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*");DATIVE_PREP=re.compile(r"\b(?:благодаря|согласно|вопреки)\b",re.IGNORECASE)
DOC_RE=re.compile(r"\bв\s+(?P<target>раздел|подраздел|пункт|подпункт|графу|таблицу)(?P<number>\s+(?:№\s*)?\d+)\b",re.IGNORECASE);DOC_FORMS={'раздел':'разделе','подраздел':'подразделе','пункт':'пункте','подпункт':'подпункте','графу':'графе','таблицу':'таблице'}
STATIVE_RE=re.compile(r"\b(?:приведен[аоы]?|указан[аоы]?|отражен[аоы]?|содержится|содержатся|зафиксирован[аоы]?|представлен[аоы]?)\b",re.IGNORECASE)
NUMERAL_RE=re.compile(r"\b(?P<predicate>[А-Яа-яЁё-]+)\s+(?:два|две|три|четыре)\s+[А-Яа-яЁё-]+\b",re.IGNORECASE);AUX_RE=re.compile(r"\b(?P<aux>был|была|было)\s+(?P<predicate>[А-Яа-яЁё-]+)\b",re.IGNORECASE)
PROCESS_COORD_RE=re.compile(r"\bпри\s+(?P<first>[А-Яа-яЁё-]+)(?:\s+[А-Яа-яЁё-]+){1,4}\s+и\s+(?P<second>[А-Яа-яЁё-]+)\b",re.IGNORECASE)
ORDER_PROCESS_RE=re.compile(r"\bпоряд(?:ок|ка|ку|ком|ке|ки|ков|кам|ками|ках)\s+(?P<process>[А-Яа-яЁё-]+)\s+(?P<object>[А-Яа-яЁё-]+)\b",re.IGNORECASE)
REPORTING_LEMMAS={'выявить','обнаружить','установить','зафиксировать','зарегистрировать'};PROCESS_SUFFIXES=('ция','ение','ание','тие','ство');PREPOSITIONS={'в','во','на','по','с','со','из','от','у','для','при','к','ко','о','об','под','над','между','через'}
def _render(word,parse,grams):
 try:form=parse.inflect(grams)
 except Exception:form=None
 if not form or not form.word:return None
 fixed=preserve_capitalization(word,preserve_yo(word,form.word));return fixed if fixed.casefold()!=word.casefold() else None
def _from_parses(word,parses,grams):
 forms={fixed for parse in parses if (fixed:=_render(word,parse,grams))};return next(iter(forms)) if len(forms)==1 else None
def _best_form(word,parses,grams):
 """Prefer common lexical parses over name/homonym readings in a governed frame."""
 ranked=sorted(parses,key=lambda p:(any(mark in p.tag for mark in ('Name','Surn','Patr','Geox')), -float(getattr(p,'score',0.0))))
 for parse in ranked:
  fixed=_render(word,parse,grams)
  if fixed:return fixed
 return None
class GovernmentFrameStage:
 def __init__(self,morphology=None):self.morph=morphology or get_morphology();self.calls=0;self.found={'coordination':0,'dative':0,'po':0,'locative':0,'numeral':0,'process_coordination':0,'order_process':0,'auxiliary':0}
 def _coordination(self,text):
  words=list(WORD_RE.finditer(text));out=[]
  for i in range(2,len(words)-1):
   if words[i].group().casefold()!='и':continue
   governor,first,second=words[i-2],words[i-1],words[i+1]
   if not all(text[a.end():b.start()].isspace() for a,b in ((governor,first),(first,words[i]),(words[i],second))):continue
   first_n=self.morph.noun_parses(first.group());second_n=self.morph.noun_parses(second.group());gov_n=self.morph.noun_parses(governor.group())
   if not first_n or not second_n or not any(p.tag.case=='gent' for p in gov_n):continue
   targets={(p.tag.case,p.tag.number) for p in second_n if p.tag.case=='gent' and p.tag.number};lemmas1=self.morph.lemmas(first.group());lemmas2=self.morph.lemmas(second.group())
   if not any(x.endswith(PROCESS_SUFFIXES) for x in lemmas1) or not any(x.endswith(PROCESS_SUFFIXES) for x in lemmas2):continue
   forms={_from_parses(first.group(),first_n,{case,number}) for case,number in targets};forms.discard(None)
   if len(forms)==1:out.append(EditCandidate(first.group(),forms.pop(),.997,'rule-government-coordination','Однородные названия функций после родительного падежа должны иметь одинаковую форму.',start=first.start(),sources=('rule-government-coordination',)));self.found['coordination']+=1
  return out
 def _dative(self,text):
  words=list(WORD_RE.finditer(text));out=[]
  for prep in DATIVE_PREP.finditer(text):
   start_idx=next((i for i,w in enumerate(words) if w.start()>=prep.end()),None)
   if start_idx is None:continue
   window=[]
   for w in words[start_idx:start_idx+7]:
    gap=text[prep.end() if not window else window[-1].end():w.start()]
    if re.search(r'[,.;:!?()]',gap):break
    window.append(w)
   head=None;head_parses=[]
   for w in window:
    nouns=self.morph.noun_parses(w.group())
    if not nouns:continue
    gent=[p for p in nouns if p.tag.case=='gent' and p.tag.number]
    if gent:head=w;head_parses=gent;break
    if any(p.tag.case=='datv' for p in nouns):head=None;break
    if not any(p.tag.case=='ablt' for p in nouns):break
   if head is None:continue
   preferred=sorted(head_parses,key=lambda p:(any(mark in p.tag for mark in ('Name','Surn','Patr','Geox')),-float(getattr(p,'score',0.0))))
   if not preferred or not preferred[0].tag.number:continue
   number=preferred[0].tag.number;fixed_head=_best_form(head.group(),[p for p in preferred if p.tag.number==number],{'datv',number})
   if fixed_head:out.append(EditCandidate(head.group(),fixed_head,.998,'rule-dative-frame',f'Предлог «{prep.group()}» требует дательного падежа.',start=head.start(),sources=('rule-dative-frame',)))
   modifier_added=False
   for w in window:
    if w.start()>=head.start():break
    parses=[p for p in self.morph.attributive_parses(w.group()) if p.tag.case=='gent' and p.tag.number==number];fixed=_best_form(w.group(),parses,{'datv',number}) if parses else None
    if fixed:out.append(EditCandidate(w.group(),fixed,.998,'rule-dative-frame',f'Определение после «{prep.group()}» согласуется с существительным в дательном падеже.',start=w.start(),sources=('rule-dative-frame',)));modifier_added=True
   if fixed_head or modifier_added:self.found['dative']+=1
  return out
 def _po(self,text):
  out=[];pattern=re.compile(r"\bпо\s+(?P<modifier>[А-Яа-яЁё-]+)\s+(?P<head>[А-Яа-яЁё-]+)\b",re.IGNORECASE)
  for m in pattern.finditer(text):
   mods=self.morph.attributive_parses(m.group('modifier'));heads=self.morph.noun_parses(m.group('head'));source=[p for p in heads if p.tag.case=='gent' and p.tag.number=='plur']
   if not any(p.tag.case in {'datv','loct'} and p.tag.number=='sing' for p in mods) or not source:continue
   fixed=_from_parses(m.group('head'),source,{'datv','sing'})
   if fixed:out.append(EditCandidate(m.group('head'),fixed,.996,'rule-po-government','После «по» название направления подготовки употребляется в дательном падеже.',start=m.start('head'),sources=('rule-po-government',)));self.found['po']+=1
  return out
 def _process_coordination(self,text):
  out=[]
  for m in PROCESS_COORD_RE.finditer(text):
   for name in ('first','second'):
    word=m.group(name);parses=[p for p in self.morph.noun_parses(word) if p.normal_form.endswith(PROCESS_SUFFIXES)]
    if not parses:continue
    fixed=_best_form(word,parses,{'loct','sing'})
    if fixed:out.append(EditCandidate(word,fixed,.997,'rule-pri-process-coordination','Однородные названия процессов после «при» употребляются в предложном единственного числа.',start=m.start(name),sources=('rule-pri-process-coordination',)));self.found['process_coordination']+=1
  return out
 def _order_process(self,text):
  out=[]
  for m in ORDER_PROCESS_RE.finditer(text):
   word=m.group('process');parses=[p for p in self.morph.noun_parses(word) if p.normal_form.endswith(PROCESS_SUFFIXES)]
   if not parses:continue
   fixed=_best_form(word,parses,{'gent','sing'})
   if fixed:out.append(EditCandidate(word,fixed,.998,'rule-order-process','После слова «порядок» название процесса употребляется в родительном единственного числа.',start=m.start('process'),sources=('rule-order-process',)));self.found['order_process']+=1
  return out
 def _locative(self,text):
  out=[]
  for m in DOC_RE.finditer(text):
   if not STATIVE_RE.search(text[max(0,m.start()-180):m.start()]):continue
   source=m.group('target');fixed=preserve_capitalization(source,DOC_FORMS[source.casefold()]);out.append(EditCandidate(source,fixed,.998,'rule-document-locative','Указание места в структуре документа требует предложного падежа.',start=m.start('target'),sources=('rule-document-locative',)));self.found['locative']+=1
  return out
 def _numeral(self,text):
  out=[]
  for m in NUMERAL_RE.finditer(text):
   word=m.group('predicate');parses=[p for p in self.morph.known_parses(word) if p.tag.POS=='PRTS' and p.tag.number=='sing' and p.normal_form in REPORTING_LEMMAS];fixed=_from_parses(word,parses,{'plur'}) if parses else None
   if fixed:out.append(EditCandidate(word,fixed,.996,'rule-numeral-predicate','Сказуемое согласуется с количественной группой во множественном числе.',start=m.start('predicate'),sources=('rule-numeral-predicate',)));self.found['numeral']+=1
  return out
 def _auxiliary(self,text):
  words=list(WORD_RE.finditer(text));out=[]
  for m in AUX_RE.finditer(text):
   pred=[p for p in self.morph.known_parses(m.group('predicate')) if p.tag.POS in {'PRTS','ADJS'} and p.tag.number=='plur']
   if not pred:continue
   aux_idx=next((i for i,w in enumerate(words) if w.start()==m.start('aux')),None);subject=None
   if aux_idx is None:continue
   for i in range(aux_idx-1,max(-1,aux_idx-12),-1):
    if re.search(r'[.!?;]',text[words[i].end():m.start('aux')]):break
    if i>0 and words[i-1].group().casefold() in PREPOSITIONS:continue
    nouns=[p for p in self.morph.noun_parses(words[i].group()) if p.tag.case=='nomn' and p.tag.number=='plur']
    if nouns:subject=words[i];break
   if subject:out.append(EditCandidate(m.group('aux'),preserve_capitalization(m.group('aux'),'были'),.997,'rule-auxiliary-agreement',f'Связка согласуется с подлежащим «{subject.group()}» во множественном числе.',start=m.start('aux'),sources=('rule-auxiliary-agreement',)));self.found['auxiliary']+=1
  return out
 def candidates(self,text):
  self.calls+=1
  if not self.morph.available:return []
  return self._coordination(text)+self._dative(text)+self._po(text)+self._process_coordination(text)+self._order_process(text)+self._locative(text)+self._numeral(text)+self._auxiliary(text)
 def metrics(self):return {'calls':self.calls,**self.found}
