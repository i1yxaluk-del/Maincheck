"""Production rejection policy for unsupported model and rule edits."""
from __future__ import annotations
import re
from coverage_policy import is_punctuation_only
from verification import is_generative
from morphology import get_morphology
ACRONYM=re.compile(r'\b[А-ЯЁA-Z]{2,12}\b');DIGITS=re.compile(r'\d+');QUOTES=re.compile(r'[«»„“”"]');DELIMITERS='()[]{}';PROTECTED_NOTATION=re.compile(r'\b[А-ЯЁA-Z](?:\([А-ЯЁA-Z]\))?[А-ЯЁA-Z]{1,12}\b');WORD_RE=re.compile(r'[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*');NUMERAL_PREFIX=re.compile(r'(?:\b[2-4]|\b(?:два|две|три|четыре))\s+(?:[А-Яа-яЁё-]+\s+){0,2}$',re.IGNORECASE);DOUBLE_INITIAL=re.compile(r'(?<![А-Яа-яЁё])([А-Яа-яЁё])\1[А-Яа-яЁё]{3,}',re.IGNORECASE);EMBEDDED_CAPITAL=re.compile(r'[а-яё][А-ЯЁ][А-Яа-яЁё]{2,}')
class ProductionSafety:
 def __init__(self):self.morph=get_morphology();self.counters={'digits':0,'acronym':0,'quote_style':0,'delimiter':0,'protected_notation':0,'parenthesis_punctuation':0,'tainted_sentence':0,'existing_np_agreement':0,'quoted':0,'enumeration':0,'solo_draft':0,'word_boundary':0,'novel_repetition':0,'numeral_agreement':0}
 @staticmethod
 def _start(text,c):
  if c.start is not None and 0<=c.start<=len(text)-len(c.before) and text[c.start:c.start+len(c.before)]==c.before:return c.start
  found=[m.start() for m in re.finditer(re.escape(c.before),text)];return found[0] if len(found)==1 else None
 @staticmethod
 def _inside_quotes(text,pos):return pos is not None and text.rfind('«',0,pos+1)>text.rfind('»',0,pos+1)
 @staticmethod
 def _family(c):
  names=getattr(c,'sources',()) or (c.category,)
  if any(x.startswith('draft') for x in names):return 'draft'
  if any(x.startswith('sage') for x in names):return 'sage'
  if any(x.startswith('russian-gec') for x in names):return 'gec'
  return c.category.split('-',1)[0]
 @staticmethod
 def _sentence(text,pos):
  if pos is None:return (0,len(text))
  left=max(text.rfind('.',0,pos),text.rfind('!',0,pos),text.rfind('?',0,pos),text.rfind('\n\n',0,pos));right=[x for x in (text.find('.',pos),text.find('!',pos),text.find('?',pos),text.find('\n\n',pos)) if x>=0];return (left+1,min(right)+1 if right else len(text))
 @staticmethod
 def _delimiter_changed(before,after):return any(before.count(ch)!=after.count(ch) for ch in DELIMITERS)
 @staticmethod
 def _parenthesis_punctuation(text,c,pos):
  if any(bool(re.search(p,c.after)) and not re.search(p,c.before) for p in (r',\s*\(',r'\)\s*,',r';\s*\(',r'\)\s*;')):return True
  if pos is None:return False
  inserted_comma=c.after.count(',')>c.before.count(',');removed_comma=c.after.count(',')<c.before.count(',');right=text[pos+len(c.before):];left=text[:pos]
  if inserted_comma and (re.match(r'\s*\(',right) or re.search(r'\)\s*$',left)):return True
  if removed_comma and (re.match(r'\s*\(',right) or re.search(r'\)\s*$',left)):return True
  return False
 def _already_agrees_with_following_noun(self,text,c,pos):
  if not c.category.startswith(('rule-agreement','rule-copular-agreement')) or pos is None or not re.fullmatch(WORD_RE,c.before):return False
  words=list(WORD_RE.finditer(text,pos+len(c.before)))
  for w in words[:3]:
   gap=text[pos+len(c.before):w.start()]
   if re.search(r'[,.;:!?()]',gap):break
   if self.morph.noun_parses(w.group()) and self.morph.pair_agrees(c.before,w.group()):return True
  return False
 def _valid_numeral_form_changed(self,text,c,pos):
  if pos is None or not c.category.startswith('rule-agreement') or not re.fullmatch(WORD_RE,c.before) or not re.fullmatch(WORD_RE,c.after):return False
  if not NUMERAL_PREFIX.search(text[max(0,pos-48):pos]):return False
  src=self.morph.known_parses(c.before);dst=self.morph.known_parses(c.after)
  if not src or not dst:return False
  src_valid=any((p.tag.POS in {'ADJF','PRTF'} and p.tag.case=='gent' and p.tag.number=='plur') or (p.tag.POS=='NOUN' and p.tag.case=='gent' and p.tag.number=='sing') for p in src)
  if not src_valid:return False
  src_numbers={p.tag.number for p in src if p.tag.number};dst_numbers={p.tag.number for p in dst if p.tag.number}
  return bool(src_numbers and dst_numbers and src_numbers.isdisjoint(dst_numbers))
 def _structural_reason(self,text,c,pos):
  if pos is None or not is_generative(c.category) or is_punctuation_only(c):return ''
  a,b=self._sentence(text,pos);segment=text[a:b];local=pos-a;patched=segment[:local]+c.after+segment[local+len(c.before):]
  if len(WORD_RE.findall(segment))!=len(WORD_RE.findall(patched)):return 'word_boundary'
  if len(DOUBLE_INITIAL.findall(patched))>len(DOUBLE_INITIAL.findall(segment)):return 'novel_repetition'
  if len(EMBEDDED_CAPITAL.findall(patched))>len(EMBEDDED_CAPITAL.findall(segment)):return 'word_boundary'
  return ''
 def _reason(self,text,c):
  if DIGITS.findall(c.before)!=DIGITS.findall(c.after):return 'digits'
  if ACRONYM.findall(c.before)!=ACRONYM.findall(c.after) and not c.category.startswith('rule-rag-terminology'):return 'acronym'
  if PROTECTED_NOTATION.findall(c.before)!=PROTECTED_NOTATION.findall(c.after) and not c.category.startswith('rule-rag-terminology'):return 'protected_notation'
  if QUOTES.findall(c.before)!=QUOTES.findall(c.after):return 'quote_style'
  if self._delimiter_changed(c.before,c.after):return 'delimiter'
  generative=is_generative(c.category);punct=is_punctuation_only(c);pos=self._start(text,c)
  if self._valid_numeral_form_changed(text,c,pos):return 'numeral_agreement'
  if self._already_agrees_with_following_noun(text,c,pos):return 'existing_np_agreement'
  structural=self._structural_reason(text,c,pos)
  if structural:return structural
  if generative and punct and self._parenthesis_punctuation(text,c,pos):return 'parenthesis_punctuation'
  if generative and not punct and self._inside_quotes(text,pos):return 'quoted'
  if pos is not None:
   stop=text.find('\n',pos);line=text[text.rfind('\n',0,pos)+1:stop if stop>=0 else len(text)]
   if re.match(r'^\s*\d+[.)]\s+',line) and not c.category.startswith('rule-'):return 'enumeration'
  if generative and not punct and c.category.startswith('draft') and getattr(c,'votes',1)<2:return 'solo_draft'
  return ''
 def allowed(self,text,c):
  reason=self._reason(text,c)
  if reason:self.counters[reason]+=1;return False
  return True
 def filter(self,text,candidates):
  accepted=[];tainted=[];structural={'delimiter','protected_notation','parenthesis_punctuation','quote_style','word_boundary','novel_repetition'}
  for c in candidates:
   reason=self._reason(text,c);pos=self._start(text,c)
   if reason:
    self.counters[reason]+=1
    if is_generative(c.category) and reason in structural:tainted.append((*self._sentence(text,pos),self._family(c)))
   else:accepted.append(c)
  result=[]
  for c in accepted:
   pos=self._start(text,c);family=self._family(c)
   if is_generative(c.category) and pos is not None and any(a<=pos<b and family==f for a,b,f in tainted):self.counters['tainted_sentence']+=1;continue
   result.append(c)
  return result
 def metrics(self):return dict(self.counters)
