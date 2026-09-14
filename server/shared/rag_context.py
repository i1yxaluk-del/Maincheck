"""Общий безопасный RAG-контекст для локального и сетевого маршрутов."""
from __future__ import annotations
import logging,os
from .rag_store import HashingEmbedder,OllamaEmbedder,RagStore
_log=logging.getLogger("ai_suggester.rag.context")
_store=_embedder=None

def context_for(text:str)->str:
 global _store,_embedder
 if os.getenv("RAG_ENABLED","false").lower() not in {"1","true","yes","on"}:return ""
 try:
  if _store is None:
   _store=RagStore(os.getenv("RAG_STORE_DIR","data/rag_store"))
   if os.getenv("RAG_EMBEDDER","ollama")=="hashing":_embedder=HashingEmbedder(int(os.getenv("RAG_HASHING_DIM","1024")))
   else:_embedder=OllamaEmbedder(os.getenv("RAG_EMBED_MODEL","nomic-embed-text"),os.getenv("OLLAMA_URL","http://localhost:11434"))
  hits=_store.search(text,top_k=int(os.getenv("RAG_TOP_K","6")),embedder=_embedder)
  if not hits:return ""
  lines=["ВЕДОМСТВЕННЫЕ И НОРМАТИВНЫЕ ФРАГМЕНТЫ. Используй только когда применимы; не изменяй факты и обозначения:"]
  lines.extend(f"— [{h['doc_id']}, версия {h['version']}, фрагмент {h['chunk_id']}] {h['text'][:700]}" for h in hits)
  return "\n".join(lines)
 except Exception as exc:_log.warning("RAG недоступен, продолжаю без него: %s",exc);return ""
