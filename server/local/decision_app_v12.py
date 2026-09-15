"""v12: нормативный RAG и доказуемые терминологические правки поверх v11."""
from decision_app_v11 import app,router  # noqa: F401
from decision_app import logger
from shared.rag_context import context_for,last_evidence
from rag_terminology import RagTerminologyStage
import reasoning_cascade
from hybrid_editor import OllamaDraftClient

_NORMATIVE_POLICY=("\nЕсли в контексте есть НОРМАТИВНЫЕ ДОКАЗАТЕЛЬСТВА: различай правило, определение, пример и область применения. Применяй норму только когда совпадают объект и условие. Канонические сокращённые наименования исправляй точно; не придумывай нормы, которых нет в источнике.")
if _NORMATIVE_POLICY not in reasoning_cascade.REASONER_SYSTEM:
    reasoning_cascade.REASONER_SYSTEM+=_NORMATIVE_POLICY
if _NORMATIVE_POLICY not in OllamaDraftClient.SYSTEM:
    OllamaDraftClient.SYSTEM+=_NORMATIVE_POLICY

_terminology=RagTerminologyStage();_previous_candidates=router.candidates
async def _rag_candidates(text:str,context:str):
    rag=context_for(text);enriched=(context+'\n\n'+rag).strip() if rag else context
    base=await _previous_candidates(text,enriched);terms=_terminology.candidates(text)
    evidence=last_evidence()
    if rag or terms:logger.info('RAG local evidence=%d context_chars=%d terminology_candidates=%d',len(evidence),len(rag),len(terms))
    return router.arbiter.merge(base+terms)
router.candidates=_rag_candidates
_previous_metrics=router.metrics
def _rag_metrics():
    result=_previous_metrics();result['rag_terminology']=_terminology.metrics();result['rag_last_evidence']=len(last_evidence());return result
router.metrics=_rag_metrics
app.version='12.2'
