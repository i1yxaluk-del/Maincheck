"""v12: нормативный RAG с ограничением интерактивной задержки."""
import asyncio,os
import decision_app_v11 as v11
from decision_app_v11 import app,router  # noqa: F401
from decision_app import logger
from shared.rag_context import context_for,last_evidence
from rag_terminology import RagTerminologyStage
from v16_safety import suppress_unsafe_solo_punctuation
from client_safe_edits import materialize_client_safe_deletions
import reasoning_cascade
from hybrid_editor import OllamaDraftClient
_POLICY=("\nНОРМАТИВНЫЕ ДОКАЗАТЕЛЬСТВА: различай правило, определение, пример и область применения. Применяй норму только при совпадении объекта и условия. Канонические наименования исправляй точно; не выдумывай нормы.")
if _POLICY not in reasoning_cascade.REASONER_SYSTEM:reasoning_cascade.REASONER_SYSTEM+=_POLICY
if _POLICY not in OllamaDraftClient.SYSTEM:OllamaDraftClient.SYSTEM+=_POLICY
_terminology=RagTerminologyStage();_previous=router.candidates
async def _fast(text,context):
 fn=getattr(v11,'_fast_candidates',None)
 if fn is None:return await _previous(text,context)
 found=await fn(text,context);found=suppress_unsafe_solo_punctuation(found,text);return materialize_client_safe_deletions(text,found)
async def _rag_candidates(text,context):
 # Дешёвые доказуемые термины проверяются до DeepSeek.
 terms=_terminology.candidates(text)
 if terms:
  base=await _fast(text,context);logger.info('RAG fast-path terminology_candidates=%d',len(terms));return router.arbiter.merge(base+terms)
 rag=await asyncio.to_thread(context_for,text);enriched=(context+'\n\n'+rag).strip() if rag else context
 timeout=float(os.getenv('REQUEST_PIPELINE_TIMEOUT','30'))
 try:base=await asyncio.wait_for(_previous(text,enriched),timeout=timeout)
 except asyncio.TimeoutError:
  logger.warning('RAG/pipeline timeout %.1fs; возврат к быстрому конвейеру',timeout);base=await _fast(text,enriched)
 evidence=last_evidence()
 if rag:logger.info('RAG local evidence=%d context_chars=%d',len(evidence),len(rag))
 return router.arbiter.merge(base)
router.candidates=_rag_candidates
_old_metrics=router.metrics
def _metrics():
 r=_old_metrics();r['rag_terminology']=_terminology.metrics();r['rag_last_evidence']=len(last_evidence());r['request_pipeline_timeout']=float(os.getenv('REQUEST_PIPELINE_TIMEOUT','30'));return r
router.metrics=_metrics;app.version='12.3'
