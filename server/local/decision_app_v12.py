"""v12: нормативный RAG и paragraph-aware модели с ограниченной задержкой."""
import asyncio,os
import decision_app_v11 as v11
from decision_app_v11 import app,router  # noqa: F401
from decision_app import logger
from shared.rag_context import context_for,last_evidence_count,should_use_rag
from rag_terminology import RagTerminologyStage
from v16_safety import suppress_unsafe_solo_punctuation
from client_safe_edits import materialize_client_safe_deletions
from punctuation_pipeline import collapse_soft_breaks,restore_soft_breaks
from logical_segmentation import split_logical_sentences
import hybrid_editor,reasoning_cascade,punctuation_pipeline
from hybrid_editor import OllamaDraftClient
# Все модели получают логическое предложение целиком; одинарный \n — визуальный перенос.
hybrid_editor.split_sentences=split_logical_sentences
reasoning_cascade.split_sentences=split_logical_sentences
punctuation_pipeline.split_sentences=split_logical_sentences
# SAGE/GEC видят строку без визуальных переносов, после вывода исходные переносы возвращаются.
if getattr(router,'sage',None) is not None and not getattr(router.sage,'_layout_wrapped',False):
 _sage_batch=router.sage.correct_batch
 async def _sage_layout(texts):
  flat=[collapse_soft_breaks(x) for x in texts];out=await _sage_batch(flat)
  return [restore_soft_breaks(src,dst) for src,dst in zip(texts,out)]
 router.sage.correct_batch=_sage_layout;router.sage._layout_wrapped=True
if getattr(router,'gec',None) is not None and not getattr(router.gec,'_layout_wrapped',False):
 _gec_correct=router.gec.correct
 async def _gec_layout(text):return restore_soft_breaks(text,await _gec_correct(collapse_soft_breaks(text)))
 router.gec.correct=_gec_layout;router.gec._layout_wrapped=True
_POLICY=("\nНОРМАТИВНЫЕ ДОКАЗАТЕЛЬСТВА: различай правило, определение, пример и область применения. Применяй норму только при совпадении объекта и условия. Канонические наименования исправляй точно; не выдумывай нормы.")
if _POLICY not in reasoning_cascade.REASONER_SYSTEM:reasoning_cascade.REASONER_SYSTEM+=_POLICY
if _POLICY not in OllamaDraftClient.SYSTEM:OllamaDraftClient.SYSTEM+=_POLICY
# Ограничиваем только reasoning; быстрые кандидаты не запускаются второй раз после timeout.
if getattr(v11,'_cascade',None) is not None and not getattr(v11._cascade,'_total_timeout_wrapped',False):
 _cascade_candidates=v11._cascade.candidates
 async def _bounded_cascade(text,context=''):
  limit=float(os.getenv('REASONING_TOTAL_TIMEOUT','25'))
  try:return await asyncio.wait_for(_cascade_candidates(text,context),timeout=limit)
  except asyncio.TimeoutError:logger.warning('Reasoning total timeout %.1fs; используются уже рассчитанные быстрые кандидаты',limit);return []
 v11._cascade.candidates=_bounded_cascade;v11._cascade._total_timeout_wrapped=True
_terminology=RagTerminologyStage();_previous=router.candidates
async def _fast(text,context):
 fn=getattr(v11,'_fast_candidates',None)
 if fn is None:return await _previous(text,context)
 found=await fn(text,context);found=suppress_unsafe_solo_punctuation(found,text);return materialize_client_safe_deletions(text,found)
async def _candidates(text,context):
 terms=_terminology.candidates(text)
 if terms:
  base=await _fast(text,context);logger.info('RAG fast-path terminology_candidates=%d',len(terms));return router.arbiter.merge(base+terms)
 rag=await asyncio.to_thread(context_for,text) if should_use_rag(text) else ''
 enriched=(context+'\n\n'+rag).strip() if rag else context;base=await _previous(text,enriched)
 if rag:logger.info('RAG local evidence=%d context_chars=%d',last_evidence_count(),len(rag))
 return router.arbiter.merge(base)
router.candidates=_candidates
_old_metrics=router.metrics
def _metrics():
 r=_old_metrics();r['rag_terminology']=_terminology.metrics();r['rag_last_evidence']=last_evidence_count();r['reasoning_total_timeout']=float(os.getenv('REASONING_TOTAL_TIMEOUT','25'));return r
router.metrics=_metrics;app.version='12.4'
