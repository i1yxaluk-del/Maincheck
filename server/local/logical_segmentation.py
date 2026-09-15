"""Сегментация по логическим предложениям, а не по визуальным переносам строк."""
from __future__ import annotations
import re
from segmentation import Segment,_is_hard_break
_END=re.compile(r"([.!?;…]+|\n{2,})(\s+|$)")
def split_logical_sentences(text:str,min_length:int=1)->list[Segment]:
 if not text:return []
 result=[];cursor=0
 for match in _END.finditer(text):
  if not _is_hard_break(text,match,cursor):continue
  end=match.end(1);raw=text[cursor:end];stripped=raw.strip()
  if len(stripped)>=min_length:
   result.append(Segment(stripped,cursor+len(raw)-len(raw.lstrip())))
  cursor=match.end()
 tail=text[cursor:]
 if tail.strip():result.append(Segment(tail.strip(),cursor+len(tail)-len(tail.lstrip())))
 return result or [Segment(text.strip(),len(text)-len(text.lstrip()))]
