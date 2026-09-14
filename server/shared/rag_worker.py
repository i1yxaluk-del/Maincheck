"""Фоновая служба инкрементальной индексации RAG."""
from __future__ import annotations
import fcntl,json,logging,os,signal,time
from pathlib import Path
from .rag_store import HashingEmbedder,OllamaEmbedder,RagStore

logging.basicConfig(level=os.getenv('LOG_LEVEL','INFO'),format='%(asctime)s %(levelname)s %(message)s')
log=logging.getLogger('ai_suggester.rag.worker');running=True

def _stop(*_):
 global running;running=False

def _write_status(path,payload):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)

def main():
 global running
 documents=Path(os.getenv('RAG_DOCUMENTS_DIR','/home/service/llama/RAG/documents'))
 state=Path(os.getenv('RAG_STORE_DIR','/home/service/llama/RAG/state'));state.mkdir(parents=True,exist_ok=True)
 interval=max(5,int(os.getenv('RAG_WORKER_INTERVAL','30')));settle=max(0,int(os.getenv('RAG_SETTLE_SECONDS','10')))
 chunk=int(os.getenv('RAG_CHUNK_CHARS','600'));overlap=int(os.getenv('RAG_CHUNK_OVERLAP','80'))
 prune=os.getenv('RAG_AUTO_PRUNE','false').lower() in {'1','true','yes','on'}
 lock=(state/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 store=RagStore(state)
 embedder=HashingEmbedder(int(os.getenv('RAG_HASHING_DIM','1024'))) if os.getenv('RAG_EMBEDDER','ollama')=='hashing' else OllamaEmbedder(os.getenv('RAG_EMBED_MODEL','nomic-embed-text'),os.getenv('OLLAMA_URL','http://localhost:11434'))
 signal.signal(signal.SIGTERM,_stop);signal.signal(signal.SIGINT,_stop)
 log.info('RAG worker: documents=%s state=%s interval=%ss',documents,state,interval)
 while running:
  started=time.time()
  try:
   plan=store.scan_folder(documents);changes=[x for x in plan if x['state']!='unchanged']
   if changes and settle:
    time.sleep(settle)
   result=store.sync_folder(documents,embedder=embedder,chunk_chars=chunk,overlap=overlap,prune=prune) if changes else {'unchanged':len(plan),'errors':[]}
   payload={'ok':not result.get('errors'),'checked_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'duration_seconds':round(time.time()-started,2),'result':result}
   _write_status(state/'worker-status.json',payload)
   if changes:log.info('Синхронизация: %s',result)
  except Exception as error:
   log.exception('Цикл RAG завершился ошибкой');_write_status(state/'worker-status.json',{'ok':False,'checked_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'error':str(error)})
  deadline=time.time()+interval
  while running and time.time()<deadline:time.sleep(1)
 return 0
if __name__=='__main__':raise SystemExit(main())
