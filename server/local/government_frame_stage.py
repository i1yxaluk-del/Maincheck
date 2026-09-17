"""Morphology-backed government and agreement frames for official Russian."""
from __future__ import annotations
import re
from decision_engine import EditCandidate
from morphology import get_morphology,preserve_capitalization,preserve_yo
WORD_RE=re.compile(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*")
DATIVE_RE=re.compile(r"\b(?P<prep>благодаря|согласно|вопреки)\s+(?P<modifier>[А-Яа-яЁё-]+)\s+(?P<head>[А-Яа-яЁё-]+)\b",re.IGNORECASE)
PO_RE=re.compile(r"\bпо\s+(?P<modifier>[А-Яа-яЁё-]+)\s+(?P<head>[А-Яа-яЁё-]+)\b",re.IGNORECASE)
DOC_RE=re.compile(r"\bв\s+(?P<target>раздел|подраздел|пункт|подпункт|графу|таблицу)(?P<number>\s+(?:№\s*)?\d+)\b",re.IGNORECASE)
DOC_FORMS={'раздел':'разделе','подраздел':'подразделе','пункт':'пункте','подпункт':'подпункте','графу':'графе','таблицу':'таблице'}
STATIVE_RE=re.compile(r"\b(?:приведен[аоы]?|указан[аоы]?|отражен[аоы]?|содержится|содержатся|зафиксирован[аоы]?|представлен[аоы]?)\b",re.IGNORECASE)
NUMERAL_RE=re.compile(r"\b(?P<predicate>[А-Яа-яЁё-]+)\s+(?P<number>два|две|три|четыре)\s+(?P<head>[А-Яа-яЁё-]+)\b",re.IGNORECASE)
REPORTING_LEMMAS={'выявить','обнаружить','установить','зафиксировать','зарегистрировать'}
PROCESS_SUFFIXES=('ция','ение','ание','тие','ство')

def _form(morph,word,grams):
 forms={preserve_capitalization(word,preserve_yo(word,x)) for x in morph.inflected_forms(word,grams)};forms={x for x in forms if x.casefold()!=word.casefold()};return next(iter(forms)) if len(forms)==1 else None
class GovernmentFrameStage:
 def __init__(self,morphology=None):self.morph=morphology or get_morphology();self.calls=0;self.found={'coordination':0,'dative':0,'po':0,'locative':0,'numeral':0}
 def _coordination(self,text):
  words=list(WORD_RE.finditer(text));out=[]
  for i in range(2,len(words)-1):
   if words[i].group().casefold()!='и':continue
   governor,first,second=words[i-2],words[i-1],words[i+1]
   if not all(text[a.end():b.start()].isspace() for a,b in ((governor,first),(first,words[i]),(words[i],second))):continue
   first_n=self.morph.noun_parses(first.group());second_n=self.morph.noun_parses(second.group());gov_n=self.morph.noun_parses(governor.group())
   if not first_n or not second_n or not gov_n:continue
   if not any(p.tag.case=='gent' for p in gov_n):continue
   targets={(p.tag.case,p.tag.number) for p in second_n if p.tag.case=='gent' and p.tag.number}
   lemmas1=self.morph.lemmas(first.group());lemmas2=self.morph.lemmas(second.group())
   if not any(x.endswith(PROCESS_SUFFIXES) for x in lemmas1) or not any(x.endswith(PROCESS_SUFFIXES) for x in lemmas2):continue
   forms={_form(self.morph,first.group(),{case,number}) for case,number in targets};forms.discard(None)
   if len(forms)==1:
    out.append(EditCandidate(first.group(),forms.pop(),.997,'rule-government-coordination','Однородные названия функций после родительного падежа должны иметь одинаковую форму.',start=first.start(),sources=('rule-government-coordination',)));self.found['coordination']+=1
  return out
 def _dative(self,text):
  out=[]
  for m in DATIVE_RE.finditer(text):
   head=m.group('head');modifier=m.group('modifier');heads=self.morph.noun_parses(head)
   numbers={p.tag.number for p in heads if p.tag.number and p.tag.case!='datv'}
   if len(numbers)!=1:continue
   number=next(iter(numbers));fixed_head=_form(self.morph,head,{'datv',number});fixed_modifier=_form(self.morph,modifier,{'datv',number})
   if fixed_modifier:out.append(EditCandidate(modifier,fixed_modifier,.997,'rule-dative-frame',f'Предлог «{m.group("prep")}» требует дательного падежа.',start=m.start('modifier'),sources=('rule-dative-frame',)))
   if fixed_head:out.append(EditCandidate(head,fixed_head,.997,'rule-dative-frame',f'Предлог «{m.group("prep")}» требует дательного падежа.',start=m.start('head'),sources=('rule-dative-frame',)))
   if fixed_head or fixed_modifier:self.found['dative']+=1
  return out
 def _po(self,text):
  out=[]
  for m in PO_RE.finditer(text):
   modifier=m.group('modifier');head=m.group('head');mods=self.morph.attributive_parses(modifier);heads=self.morph.noun_parses(head)
   if not any(p.tag.case in {'datv','loct'} and p.tag.number=='sing' for p in mods):continue
   if not any(p.tag.case=='gent' and p.tag.number=='plur' for p in heads):continue
   fixed=_form(self.morph,head,{'datv','sing'})
   if fixed:out.append(EditCandidate(head,fixed,.996,'rule-po-government','После «по» название направления подготовки употребляется в дательном падеже.',start=m.start('head'),sources=('rule-po-government',)));self.found['po']+=1
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
   word=m.group('predicate');parses=[p for p in self.morph.known_parses(word) if p.tag.POS=='PRTS' and p.tag.number=='sing' and p.normal_form in REPORTING_LEMMAS]
   if not parses:continue
   fixed=_form(self.morph,word,{'plur'})
   if fixed:out.append(EditCandidate(word,fixed,.996,'rule-numeral-predicate','Сказуемое согласуется с количественной группой во множественном числе.',start=m.start('predicate'),sources=('rule-numeral-predicate',)));self.found['numeral']+=1
  return out
 def candidates(self,text):
  self.calls+=1
  if not self.morph.available:return []
  return self._coordination(text)+self._dative(text)+self._po(text)+self._locative(text)+self._numeral(text)
 def metrics(self):return {'calls':self.calls,**self.found}
