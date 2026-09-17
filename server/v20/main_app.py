"""Production profile: guarded hybrid pipeline plus deterministic legal style."""
import os,re
os.environ['LLM_PRESET']='Z';os.environ.setdefault('REASONING_MODE','coverage');os.environ['REASONING_TOTAL_TIMEOUT']=os.getenv('V20_MAIN_REASONING_TIMEOUT','12');os.environ.setdefault('RAG_ENABLED','true');os.environ.setdefault('DECISION_MIN_CONFIDENCE','0.65');os.environ.setdefault('DECISION_MAX_CHANGES','8');os.environ.setdefault('ARBITER_SOLO_PENALTY','0.35')
from decision_app_v12 import app,router
import decision_app_v11 as v11
from coverage_policy import is_punctuation_only
from decision_app import logger
from legal_style_stage import LegalStyleStage
from government_frame_stage import GovernmentFrameStage
from production_safety import ProductionSafety
_SHORT_GRAMMAR_SIGNAL=re.compile(r'\b(?:был|была|было|были|остался|осталась|осталось|остались|оставался|оставалась|оставалось|оставались)\s+[А-Яа-яЁё-]+|\b[А-Яа-яЁё-]+(?:н|т)[аоы]\s+(?:два|две|три|четыре|\d+)\b',re.IGNORECASE)
_legal=LegalStyleStage();_frames=GovernmentFrameStage(router.rules.morph_helper);_safety=ProductionSafety();_rules=router.rules.candidates
router.rules.candidates=lambda text:_rules(text)+_legal.candidates(text)+_frames.candidates(text)
def _production_review(text,candidates):
 if candidates and all(c.category.startswith('rule-') for c in candidates):return False
 if candidates and any(not is_punctuation_only(c) for c in candidates):return False
 if len(text.strip())>=180:return True
 return bool(_SHORT_GRAMMAR_SIGNAL.search(text))
v11.needs_deep_review=_production_review
_base_candidates=router.candidates
async def _production_candidates(text,context=''):
 candidates=await _base_candidates(text,context);safe=[c for c in candidates if _safety.allowed(text,c)]
 if len(safe)!=len(candidates):logger.info('v20 production safety rejected=%d accepted_candidates=%d',len(candidates)-len(safe),len(safe))
 return safe
router.candidates=_production_candidates
_base_metrics=router.metrics
def _metrics():
 result=_base_metrics();result['legal_style']=_legal.metrics();result['government_frames']=_frames.metrics();result['v20_production_safety']=_safety.metrics();result['v20_reasoning_timeout']=float(os.environ['REASONING_TOTAL_TIMEOUT']);return result
router.metrics=_metrics
app.title='Maincheck production: punctuation, spelling, grammar and legal style';app.version='20-main-production.3'
