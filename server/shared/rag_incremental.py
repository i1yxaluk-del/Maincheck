"""Инкрементальная синхронизация: неизменённые файлы не парсятся и не эмбеддятся."""
from __future__ import annotations
from pathlib import Path
from .rag_store_v2 import RagStore as BaseRagStore

class IncrementalRagStore(BaseRagStore):
    def sync_folder(self,folder,*,embedder,chunk_chars=1200,overlap=150,prune=False):
        summary={k:0 for k in ('added','updated','unchanged','missing','removed')};summary['errors']=[]
        docs=self.docs
        for item in self.scan_folder(folder):
            doc_id=item['doc_id'];state=item['state']
            try:
                if state=='missing':
                    if prune:self.remove_document(doc_id);summary['removed']+=1
                    else:self.set_status(doc_id,'missing');summary['missing']+=1
                    continue
                # SHA-256 совпал и документ уже активен: не открываем парсер,
                # не обращаемся к Ollama и не меняем SQLite.
                if state=='unchanged' and doc_id in docs and docs[doc_id].status=='active':
                    summary['unchanged']+=1;continue
                meta=self.add_document(doc_id=doc_id,file_path=Path(item['path']),embedder=embedder,
                    chunk_chars=chunk_chars,overlap=overlap)
                summary[meta.operation]+=1
            except Exception as error:
                # add_document считает embeddings до транзакции: при ошибке
                # старая активная версия остаётся целой и будет повторена позже.
                summary['errors'].append({'doc_id':doc_id,'error':str(error)})
        return summary
