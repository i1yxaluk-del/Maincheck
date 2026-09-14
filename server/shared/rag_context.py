"""Нормативный контекст RAG для генеративных стадий local/cloud."""
from __future__ import annotations
import logging,os
from .rag_store import HashingEmbedder,OllamaEmbedder,RagStore
_log=logging.getLogger('ai_suggester.rag.context');_store=_embedder=None

def context_for(text:str)->str:
 global _store,_embedder
 if os.getenv('RAG_ENABLED','true').lower() not in {'1','true','yes','on'}:return ''
 try:
  if _store is None:
   _store=RagStore(os.getenv('RAG_STORE_DIR','/home/service/llama/RAG/state'))
   _embedder=HashingEmbedder(int(os.getenv('RAG_HASHING_DIM','1024'))) if os.getenv('RAG_EMBEDDER','ollama')=='hashing' else OllamaEmbedder(os.getenv('RAG_EMBED_MODEL','nomic-embed-text'),os.getenv('OLLAMA_URL','http://localhost:11434'))
  if not _store.db.execute("SELECT 1 FROM chunks c JOIN documents d USING(doc_id) WHERE d.status='active' LIMIT 1").fetchone():return ''
  hits=_store.search(text,top_k=min(3,int(os.getenv('RAG_TOP_K','6'))),embedder=_embedder)
  if not hits:return ''
  # Лучший результат помещается последним: ReasoningCascade берёт хвост контекста.
  lines=['НОРМАТИВНЫЕ ФРАГМЕНТЫ. Канонические наименования имеют приоритет:']
  for h in reversed(hits):lines.append(f"— [{h['doc_id']}#{h['chunk_id']}] {h['text'][:600]}")
  _log.info('RAG hits=%d best=%s#%s type=%s',len(hits),hits[0]['doc_id'],hits[0]['chunk_id'],hits[0].get('match_type','semantic'))
  return '\n'.join(lines)
 except Exception as exc:_log.warning('RAG недоступен, продолжаю без него: %s',exc);return ''
