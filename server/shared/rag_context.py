"""Контекстный нормативный RAG: потокобезопасный и запускаемый только по сигналу."""
from __future__ import annotations
import logging,os,re,threading
from .rag_store import HashingEmbedder,OllamaEmbedder,RagStore
from .rag_normative_context import retrieve_evidence,render_evidence
_log=logging.getLogger('ai_suggester.rag.context');_tls=threading.local();_last_count=0
_ABBR=re.compile(r"\b[А-ЯЁ]{2,12}\b")
_NORMATIVE=re.compile(r"\b(?:приказ|распоряжение|инструкция|положение|регламент|гост|федеральн(?:ый|ого)\s+закон|сокращ[её]нн(?:ое|ого)\s+наименование|обозначение|реквизит)\b",re.I)
def should_use_rag(text:str)->bool:
 # Инициалы Е.А. и обычные даты не являются основанием искать случайный закон.
 return bool(_ABBR.search(text) or _NORMATIVE.search(text) or re.search(r"«[^»\n]{3,80}»",text))
def _resources():
 if not hasattr(_tls,'store'):
  _tls.store=RagStore(os.getenv('RAG_STORE_DIR','/home/service/llama/RAG/state'))
  _tls.embedder=HashingEmbedder(int(os.getenv('RAG_HASHING_DIM','1024'))) if os.getenv('RAG_EMBEDDER','ollama')=='hashing' else OllamaEmbedder(os.getenv('RAG_EMBED_MODEL','nomic-embed-text'),os.getenv('OLLAMA_URL','http://localhost:11434'))
 return _tls.store,_tls.embedder
def context_for(text:str)->str:
 global _last_count
 if os.getenv('RAG_ENABLED','true').lower() not in {'1','true','yes','on'} or not should_use_rag(text):_last_count=0;return ''
 try:
  store,embedder=_resources()
  if not store.db.execute("SELECT 1 FROM chunks c JOIN documents d USING(doc_id) WHERE d.status='active' LIMIT 1").fetchone():_last_count=0;return ''
  hits=retrieve_evidence(store,embedder,text,top_k=min(3,int(os.getenv('RAG_TOP_K','3'))));_last_count=len(hits)
  if not hits:return ''
  best=hits[0];_log.info('RAG evidence=%d best=%s#%s type=%s score=%s',len(hits),best['doc_id'],best['chunk_id'],best.get('match_type'),best.get('evidence_score'))
  return render_evidence(hits)
 except Exception as exc:_last_count=0;_log.warning('RAG недоступен, продолжаю без него: %s',exc);return ''
def last_evidence_count():return _last_count
