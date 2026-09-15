from __future__ import annotations
import re
from dataclasses import dataclass

@dataclass(frozen=True)
class Edit:
    start:int; end:int; before:str; after:str; reason:str="model correction"; confidence:float=.75

def apply_edits(text:str, edits:list[Edit])->tuple[str,list[Edit]]:
    valid=[]
    for e in sorted(edits,key=lambda x:(x.start,x.end)):
        if e.start<0 or e.end<e.start or e.end>len(text) or text[e.start:e.end]!=e.before: continue
        if not e.after or "\n" in e.before or "\n" in e.after or len(e.before)>120 or len(e.after)>120: continue
        if valid and e.start<valid[-1].end: continue
        valid.append(e)
    out=text
    for e in reversed(valid): out=out[:e.start]+e.after+out[e.end:]
    return out,valid

def render(corrected:str,edits:list[Edit])->str:
    changes="\n".join(f"{i}. «{e.before}» → «{e.after}» | {e.reason}" for i,e in enumerate(edits,1)) or "1. Ошибок не найдено."
    return f"===CORRECTED===\n{corrected}\n===CHANGES===\n{changes}\n===END==="

def parse_corrected(payload:str)->str:
    m=re.search(r"===CORRECTED===\n(.*?)\n===CHANGES===",payload,re.S)
    if not m: raise ValueError("invalid service protocol")
    return m.group(1)
