"""Production candidate selected by the v20 benchmark: guarded hybrid pipeline."""
import os,re
os.environ.setdefault('LLM_PRESET','Z')
os.environ.setdefault('REASONING_MODE','coverage')
os.environ['REASONING_TOTAL_TIMEOUT']=os.getenv('V20_MAIN_REASONING_TIMEOUT','12')
os.environ.setdefault('RAG_ENABLED','true')
os.environ.setdefault('DECISION_MIN_CONFIDENCE','0.65')
os.environ.setdefault('DECISION_MAX_CHANGES','8')
os.environ.setdefault('ARBITER_SOLO_PENALTY','0.35')

from decision_app_v12 import app,router
import decision_app_v11 as v11
from coverage_policy import is_punctuation_only
from verification import is_generative
from decision_app import logger

_ACRONYM=re.compile(r'\b[А-ЯЁA-Z]{2,12}\b')
_DIGITS=re.compile(r'\d+')
_SHORT_GRAMMAR_SIGNAL=re.compile(
    r'\b(?:был|была|было|были|остался|осталась|осталось|остались|оставался|оставалась|оставалось|оставались)\s+[А-Яа-яЁё-]+'
    r'|\b[А-Яа-яЁё-]+(?:н|т)[аоы]\s+(?:два|две|три|четыре|\d+)\b',re.IGNORECASE)
_safety={'digits':0,'acronym':0,'quoted':0,'enumeration':0,'solo_draft':0}

def _production_review(text,candidates):
    # Short clean selections no longer pay for a 7B pass. Long paragraphs and
    # high-signal agreement constructions still receive contextual reasoning.
    if candidates and any(not is_punctuation_only(c) for c in candidates):return False
    if len(text.strip())>=180:return True
    return bool(_SHORT_GRAMMAR_SIGNAL.search(text))

v11.needs_deep_review=_production_review

def _start(text,c):
    if c.start is not None and 0<=c.start<=len(text)-len(c.before) and text[c.start:c.start+len(c.before)]==c.before:return c.start
    found=[m.start() for m in re.finditer(re.escape(c.before),text)]
    return found[0] if len(found)==1 else None

def _inside_quotes(text,pos):
    return pos is not None and text.rfind('«',0,pos+1)>text.rfind('»',0,pos+1)

def _allowed(text,c):
    if _DIGITS.findall(c.before)!=_DIGITS.findall(c.after):_safety['digits']+=1;return False
    if _ACRONYM.findall(c.before)!=_ACRONYM.findall(c.after) and not c.category.startswith('rule-rag-terminology'):_safety['acronym']+=1;return False
    generative=is_generative(c.category)
    punct=is_punctuation_only(c)
    pos=_start(text,c)
    if generative and not punct and _inside_quotes(text,pos):_safety['quoted']+=1;return False
    if generative and not punct and pos is not None:
        line=text[text.rfind('\n',0,pos)+1:text.find('\n',pos) if text.find('\n',pos)>=0 else len(text)]
        if re.match(r'^\s*\d+[.)]\s+',line):_safety['enumeration']+=1;return False
    if generative and not punct and c.category.startswith('draft') and getattr(c,'votes',1)<2:_safety['solo_draft']+=1;return False
    return True

_base_candidates=router.candidates
async def _production_candidates(text,context=''):
    candidates=await _base_candidates(text,context)
    safe=[c for c in candidates if _allowed(text,c)]
    if len(safe)!=len(candidates):logger.info('v20 production safety rejected=%d accepted_candidates=%d',len(candidates)-len(safe),len(safe))
    return safe
router.candidates=_production_candidates
_base_metrics=router.metrics
def _metrics():
    result=_base_metrics();result['v20_production_safety']=dict(_safety);result['v20_reasoning_timeout']=float(os.environ['REASONING_TOTAL_TIMEOUT']);return result
router.metrics=_metrics
app.title='Maincheck v20 main: production hybrid-syntax-rag'
app.version='20-main-production'
