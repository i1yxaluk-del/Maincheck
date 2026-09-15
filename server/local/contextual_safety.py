"""Защита синтаксически значимых запятых от одиночных модельных удалений."""
from __future__ import annotations
import re
from verification import is_deterministic
PROTECTED=re.compile(r",\s*(?:учитывая|принимая\s+во\s+внимание)\s*,\s*что\b",re.I)
def protect_contextual_commas(candidates,text):
 out=[]
 for c in candidates:
  if is_deterministic(c.category) or c.before.count(',')<=c.after.count(','):
   out.append(c);continue
  start=max(0,(c.start or 0)-20);end=min(len(text),(c.start or 0)+len(c.before)+80)
  if PROTECTED.search(text[start:end]):continue
  out.append(c)
 return out
