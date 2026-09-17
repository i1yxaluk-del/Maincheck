from __future__ import annotations

from dataclasses import dataclass
import difflib
import json
import re
from typing import Any

@dataclass(frozen=True)
class EditCandidate:
    before: str
    after: str
    confidence: float = 1.0
    category: str = "unknown"
    reason: str = ""
    start: int | None = None
    sources: tuple[str, ...] = ()
    @property
    def votes(self) -> int:return max(1,len(self.sources))

class DecisionEngine:
    def __init__(self,min_confidence:float=.55,max_changes:int=40,max_before_chars:int=180,protected_words:set[str]|None=None,guard:Any|None=None):
        self.min_confidence=min_confidence;self.max_changes=max_changes;self.max_before_chars=max_before_chars;self.protected_words={w.casefold() for w in (protected_words or set()) if w};self.guard=guard;self.rejections=[]
    @staticmethod
    def parse(payload):
        data=payload if isinstance(payload,dict) else json.loads(payload.strip().strip('`'));raw=data.get('edits',[]) if isinstance(data,dict) else [];out=[]
        for item in raw if isinstance(raw,list) else []:
            if not isinstance(item,dict):continue
            before,after=item.get('before'),item.get('after')
            if not isinstance(before,str) or not isinstance(after,str):continue
            try:confidence=float(item.get('confidence',1.0))
            except (TypeError,ValueError):confidence=0.0
            out.append(EditCandidate(before,after,max(0.0,min(1.0,confidence)),str(item.get('category','unknown')),str(item.get('reason',''))))
        return out
    def _protected(self,before):
        tokens=re.findall(r'[A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9_-]*',before);return any(t.casefold() in self.protected_words for t in tokens)
    @staticmethod
    def _changes_compound_term(before,after):
        a=re.findall(r'[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)+',before);b=re.findall(r'[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)+',after);return bool(a and a!=b)
    @staticmethod
    def _is_unverified_llm_inflection(c):
        if not c.category.startswith(('model','unknown','languagetool','surface')) or not re.fullmatch(r'[А-Яа-яЁё-]+',c.before) or not re.fullmatch(r'[А-Яа-яЁё-]+',c.after):return False
        try:
            from morphology import get_morphology
            morph=get_morphology()
            if not morph.available:return False
            before=morph.known_parses(c.before);after=morph.known_parses(c.after)
            if not before or not after:return False
            return bool({(p.normal_form,str(p.tag).split(',',1)[0]) for p in before}&{(p.normal_form,str(p.tag).split(',',1)[0]) for p in after})
        except Exception:return False
    @staticmethod
    def _splits_or_merges_word(c):
        word=r'[А-Яа-яЁёA-Za-z]+'
        return bool((re.fullmatch(word,c.before) and re.fullmatch(rf'{word} +{word}',c.after) or re.fullmatch(rf'{word} +{word}',c.before) and re.fullmatch(word,c.after)) and c.category.startswith(('model','unknown','languagetool','surface')))
    def validate(self,text,candidates):
        accepted=[];occupied=[];self.rejections=[]
        def reject(c,why):self.rejections.append((c,why))
        for c in sorted(candidates,key=lambda x:(-x.confidence,-len(x.before))):
            if len(accepted)>=self.max_changes or not c.before or c.before==c.after:continue
            if len(c.before)>self.max_before_chars or c.confidence<self.min_confidence:reject(c,'низкая уверенность или слишком длинный фрагмент');continue
            if c.before.replace('ё','е').replace('Ё','Е')==c.after.replace('ё','е').replace('Ё','Е'):reject(c,'различие только в ё/е');continue
            if self._protected(c.before) and not c.category.startswith(('rule-rag-terminology','rule-legal-repetition')):reject(c,'защищённый термин');continue
            if self._changes_compound_term(c.before,c.after):reject(c,'подмена составного термина');continue
            if self._is_unverified_llm_inflection(c):reject(c,'неподтверждённая падежная правка от модели');continue
            if self._splits_or_merges_word(c):reject(c,'изменение границы слова без доказательства');continue
            if self.guard is not None:
                allowed,why=self.guard.allow(c,text)
                if not allowed:reject(c,why);continue
            start=self._resolve_span(text,c)
            if start is None:reject(c,'фрагмент не найден однозначно');continue
            end=start+len(c.before)
            if any(not(end<=a or start>=b) for a,b in occupied):reject(c,'пересечение с принятой правкой');continue
            occupied.append((start,end));accepted.append((start,c))
        return sorted(accepted,key=lambda x:x[0],reverse=True)
    @staticmethod
    def _resolve_span(text,c):
        if c.start is not None and 0<=c.start<=len(text)-len(c.before) and text[c.start:c.start+len(c.before)]==c.before:return c.start
        positions=[m.start() for m in re.finditer(re.escape(c.before),text)];return positions[0] if len(positions)==1 else None
    def apply(self,text,candidates):
        spans=self.validate(text,candidates);result=text
        for start,c in spans:result=result[:start]+c.after+result[start+len(c.before):]
        return result,[c for _,c in reversed(spans)]
    @staticmethod
    def diff_candidates(original,corrected):
        if original==corrected:return []
        sm=difflib.SequenceMatcher(a=original,b=corrected,autojunk=False)
        return [EditCandidate(original[i1:i2],corrected[j1:j2],1.0,'diff','server diff') for tag,i1,i2,j1,j2 in sm.get_opcodes() if tag!='equal']
