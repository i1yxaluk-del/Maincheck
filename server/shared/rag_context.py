"""Контекстный нормативный RAG для local/cloud."""
from __future__ import annotations
import logging,os
from .rag_store import HashingEmbedder,OllamaEmbedder,RagStore
from .rag_normative_context import retrieve_evidence,render_evidence
_log=logging.getLogger('ai_suggester.rag.context');_store=_embedder=None;_last=[]

def context_for(text:str)->str:
 global _store,_embedder,_last
 if os.getenv('RAG_ENABLED','true').lower() not in {'1','true','yes','on'}:_last=[];return ''
 try:
  if _store is None:
   _store=RagStore(os.getenv('RAG_STORE_DIR','/home/service/llama/RAG/state'))
   _embedder=HashingEmbedder(int(os.getenv('RAG_HASHING_DIM','1024'))) if os.getenv('RAG_EMBEDDER','ollama')=='hashing' else OllamaEmbedder(os.getenv('RAG_EMBED_MODEL','nomic-embed-text'),os.getenv('OLLAMA_URL','http://localhost:11434'))
  if not _store.db.execute("SELECT 1 FROM chunks c JOIN documents d USING(doc_id) WHERE d.status='active' LIMIT 1").fetchone():_last=[];return ''
  _last=retrieve_evidence(_store,_embedder,text,top_k=min(3,int(os.getenv('RAG_TOP_K','6'))))
  if not _last:return ''
  best=_last[0];_log.info('RAG evidence=%d best=%s#%s type=%s score=%s',len(_last),best['doc_id'],best['chunk_id'],best.get('match_type'),best.get('evidence_score'))
  return render_evidence(_last)
 except Exception as exc:_last=[];_log.warning('RAG недоступен, продолжаю без него: %s',exc);return ''

def last_evidence():return list(_last)
