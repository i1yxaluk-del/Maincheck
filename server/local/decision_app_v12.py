"""v12: RAG-контекст и доказуемые терминологические правки поверх v11."""
from decision_app_v11 import app,router  # noqa: F401
from decision_app import logger
from shared.rag_context import context_for
from rag_terminology import RagTerminologyStage

_terminology=RagTerminologyStage()
_previous_candidates=router.candidates
async def _rag_candidates(text:str,context:str):
    rag=context_for(text)
    enriched=(context+'\n\n'+rag).strip() if rag else context
    base=await _previous_candidates(text,enriched)
    terms=_terminology.candidates(text)
    if rag or terms:logger.info('RAG local context_chars=%d terminology_candidates=%d',len(rag),len(terms))
    return router.arbiter.merge(base+terms)
router.candidates=_rag_candidates

_previous_metrics=router.metrics
def _rag_metrics():
    result=_previous_metrics();result['rag_terminology']=_terminology.metrics();return result
router.metrics=_rag_metrics
app.version='12.1'
