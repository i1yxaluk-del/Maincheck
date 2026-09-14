"""Точка запуска worker с общим переключателем RAG_ENABLED."""
from __future__ import annotations
import json,os,time
from pathlib import Path

def main():
    enabled=os.getenv('RAG_ENABLED','true').lower() in {'1','true','yes','on'}
    if not enabled:
        state=Path(os.getenv('RAG_STORE_DIR','/home/service/llama/RAG/state'))
        state.mkdir(parents=True,exist_ok=True)
        (state/'worker-status.json').write_text(json.dumps({'ok':True,'enabled':False,'checked_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'message':'RAG отключён через RAG_ENABLED=false'},ensure_ascii=False,indent=2),encoding='utf-8')
        return 0
    from .rag_worker import main as worker_main
    return worker_main()
if __name__=='__main__':raise SystemExit(main())
