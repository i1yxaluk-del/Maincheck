"""Production rejection policy for unsupported model edits."""
from __future__ import annotations
import re
from coverage_policy import is_punctuation_only
from verification import is_generative
ACRONYM=re.compile(r'\b[А-ЯЁA-Z]{2,12}\b');DIGITS=re.compile(r'\d+');QUOTES=re.compile(r'[«»„“”"]')
class ProductionSafety:
 def __init__(self):self.counters={'digits':0,'acronym':0,'quote_style':0,'quoted':0,'enumeration':0,'solo_draft':0}
 @staticmethod
 def _start(text,c):
  if c.start is not None and 0<=c.start<=len(text)-len(c.before) and text[c.start:c.start+len(c.before)]==c.before:return c.start
  found=[m.start() for m in re.finditer(re.escape(c.before),text)];return found[0] if len(found)==1 else None
 @staticmethod
 def _inside_quotes(text,pos):return pos is not None and text.rfind('«',0,pos+1)>text.rfind('»',0,pos+1)
 def allowed(self,text,c):
  if DIGITS.findall(c.before)!=DIGITS.findall(c.after):self.counters['digits']+=1;return False
  if ACRONYM.findall(c.before)!=ACRONYM.findall(c.after) and not c.category.startswith('rule-rag-terminology'):self.counters['acronym']+=1;return False
  if QUOTES.findall(c.before)!=QUOTES.findall(c.after):self.counters['quote_style']+=1;return False
  generative=is_generative(c.category);punct=is_punctuation_only(c);pos=self._start(text,c)
  if generative and not punct and self._inside_quotes(text,pos):self.counters['quoted']+=1;return False
  if pos is not None:
   stop=text.find('\n',pos);line=text[text.rfind('\n',0,pos)+1:stop if stop>=0 else len(text)]
   if re.match(r'^\s*\d+[.)]\s+',line) and not c.category.startswith('rule-'):
    self.counters['enumeration']+=1;return False
  if generative and not punct and c.category.startswith('draft') and getattr(c,'votes',1)<2:self.counters['solo_draft']+=1;return False
  return True
 def metrics(self):return dict(self.counters)
