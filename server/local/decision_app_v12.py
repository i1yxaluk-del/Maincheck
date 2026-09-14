"""v12: общий RAG-контекст без изменения безопасного конвейера v11."""
from decision_app_v11 import app,router  # noqa: F401
from shared.rag_context import context_for

_previous_candidates=router.candidates
async def _rag_candidates(text:str,context:str):
    rag=context_for(text)
    enriched=(context+"\n\n"+rag).strip() if rag else context
    return await _previous_candidates(text,enriched)
router.candidates=_rag_candidates
app.version="12.0"
