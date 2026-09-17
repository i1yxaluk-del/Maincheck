"""Высокоточное согласование именной части составного сказуемого."""
from __future__ import annotations
import re
from decision_engine import EditCandidate
from morphology import get_morphology,preserve_capitalization,preserve_yo
WORD_RE=re.compile(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*")
COPULAS={'быть','оставаться','являться','стать','становиться','оказаться','казаться'}
PREPOSITIONS={'в','во','на','по','с','со','из','от','у','для','при','к','ко','о','об','под','над','между','через'}
BLOCKING_POS={'PRTS','ADJS','VERB','INFN'}
class ContextualAgreementRules:
 def __init__(self,morphology=None):self.morph=morphology or get_morphology()
 @staticmethod
 def _verb_features(parses):return {(p.tag.number,p.tag.gender) for p in parses if p.tag.POS=='VERB' and p.normal_form in COPULAS}
 def candidates(self,text):
  if not self.morph.available:return []
  words=list(WORD_RE.finditer(text));out=[]
  for vi,verb in enumerate(words):
   vparses=[p for p in self.morph.known_parses(verb.group()) if p.tag.POS=='VERB' and p.normal_form in COPULAS];vf=self._verb_features(vparses)
   if not vf:continue
   subject=None
   for si in range(vi-1,max(-1,vi-15),-1):
    if re.search(r"[.!?;]",text[words[si].end():verb.start()]):break
    if si>0 and words[si-1].group().casefold() in PREPOSITIONS:continue
    nouns=[p for p in self.morph.noun_parses(words[si].group()) if p.tag.case=='nomn'];nouns=[p for p in nouns if any(p.tag.number==n and (n=='plur' or g is None or p.tag.gender==g) for n,g in vf)]
    if nouns:subject=(si,nouns);break
   if subject is None:continue
   si,nouns=subject;target_n=nouns[0].tag.number;target_g=nouns[0].tag.gender
   for ai in range(vi+1,min(len(words),vi+5)):
    gap=text[verb.end():words[ai].start()]
    if re.search(r"[.!?;,]",gap):break
    current=words[ai].group();all_parses=self.morph.known_parses(current)
    # A preposition or an already completed verbal predicate starts a new
    # dependency branch. Adjectives after it are ordinary noun modifiers,
    # not the nominal part of the copular predicate.
    if current.casefold() in PREPOSITIONS or any(p.tag.POS in BLOCKING_POS for p in all_parses):break
    parses=[p for p in self.morph.attributive_parses(current) if p.tag.POS in {'ADJF','PRTF'} and p.tag.case=='ablt']
    if not parses:continue
    if any(p.tag.number==target_n and (target_n=='plur' or p.tag.gender==target_g) for p in parses):break
    grammemes={'ablt',target_n}
    if target_n!='plur' and target_g:grammemes.add(target_g)
    forms={f.word for f in (p.inflect(grammemes) for p in parses[:8]) if f and f.word and f.word.casefold()!=current.casefold()}
    if len(forms)==1:
     fixed=preserve_capitalization(current,preserve_yo(current,forms.pop()));out.append(EditCandidate(current,fixed,.992,'rule-copular-agreement',f'именная часть сказуемого согласуется с подлежащим «{words[si].group()}»',start=words[ai].start()))
    break
  return out
