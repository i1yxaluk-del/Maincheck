"""Production rejection policy for unsupported model edits.

A model proposal is treated as a transaction inside one sentence. If it damages
brackets, protected notation, or punctuation next to parentheses, all
punctuation edits from the same model family in that sentence are discarded.
This prevents a partially accepted broken rewrite.
"""
from __future__ import annotations
import re
from coverage_policy import is_punctuation_only
from verification import is_generative
ACRONYM=re.compile(r'\b[А-ЯЁA-Z]{2,12}\b');DIGITS=re.compile(r'\d+');QUOTES=re.compile(r'[«»„“”"]')
DELIMITERS='()[]{}';PROTECTED_NOTATION=re.compile(r'\b[А-ЯЁA-Z](?:\([А-ЯЁA-Z]\))?[А-ЯЁA-Z]{1,12}\b')
class ProductionSafety:
 def __init__(self):self.counters={'digits':0,'acronym':0,'quote_style':0,'delimiter':0,'protected_notation':0,'parenthesis_punctuation':0,'tainted_sentence':0,'quoted':0,'enumeration':0,'solo_draft':0}
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
  left=max(text.rfind('.',0,pos),text.rfind('!',0,pos),text.rfind('?',0,pos),text.rfind('\n\n',0,pos));right_candidates=[x for x in (text.find('.',pos),text.find('!',pos),text.find('?',pos),text.find('\n\n',pos)) if x>=0];return (left+1,min(right_candidates)+1 if right_candidates else len(text))
 @staticmethod
 def _delimiter_changed(before,after):return any(before.count(ch)!=after.count(ch) for ch in DELIMITERS)
 @staticmethod
 def _parenthesis_punctuation(before,after):
  patterns=(r',\s*\(',r'\)\s*,',r';\s*\(',r'\)\s*;')
  return any(bool(re.search(p,after)) and not re.search(p,before) for p in patterns)
 def _reason(self,text,c):
  if DIGITS.findall(c.before)!=DIGITS.findall(c.after):return 'digits'
  if ACRONYM.findall(c.before)!=ACRONYM.findall(c.after) and not c.category.startswith('rule-rag-terminology'):return 'acronym'
  if PROTECTED_NOTATION.findall(c.before)!=PROTECTED_NOTATION.findall(c.after) and not c.category.startswith('rule-rag-terminology'):return 'protected_notation'
  if QUOTES.findall(c.before)!=QUOTES.findall(c.after):return 'quote_style'
  if self._delimiter_changed(c.before,c.after):return 'delimiter'
  generative=is_generative(c.category);punct=is_punctuation_only(c);pos=self._start(text,c)
  if generative and punct and self._parenthesis_punctuation(c.before,c.after):return 'parenthesis_punctuation'
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
  accepted=[];tainted=[]
  for c in candidates:
   reason=self._reason(text,c);pos=self._start(text,c)
   if reason:
    self.counters[reason]+=1
    if is_generative(c.category) and reason in {'delimiter','protected_notation','parenthesis_punctuation','quote_style'}:tainted.append((*self._sentence(text,pos),self._family(c)))
   else:accepted.append(c)
  result=[]
  for c in accepted:
   pos=self._start(text,c);family=self._family(c)
   if is_generative(c.category) and is_punctuation_only(c) and pos is not None and any(a<=pos<b and family==f for a,b,f in tainted):self.counters['tainted_sentence']+=1;continue
   result.append(c)
  return result
 def metrics(self):return dict(self.counters)
