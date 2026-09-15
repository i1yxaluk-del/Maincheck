"""Гибридный поиск: точное совпадение → FTS5 → семантический поиск."""
from __future__ import annotations
import re,sqlite3
from .rag_incremental import IncrementalRagStore

_TOKEN=re.compile(r"[\w\-а-яёА-ЯЁ]{2,}",re.UNICODE)

class HybridRagStore(IncrementalRagStore):
    def _result(self,row,score,kind):
        return {'score':round(score,4),'match_type':kind,'doc_id':row['doc_id'],
                'chunk_id':row['chunk_id'],'text':row['text'],
                'path':row['path'],'version':row['version']}

    def search(self,query,*,top_k=4,embedder):
        query=query.strip()
        if not query:return []
        limit=max(top_k*10,40)
        rows=list(self.db.execute("SELECT c.doc_id,c.chunk_id,c.text,d.path,d.version FROM chunks c JOIN documents d USING(doc_id) WHERE d.status='active'"))
        phrase=re.compile(r"(?<!\w)"+re.escape(query)+r"(?!\w)",re.IGNORECASE)
        exact=[r for r in rows if phrase.search(r['text'])]
        # Больше вхождений и более короткий фрагмент выше.
        exact.sort(key=lambda r:(-len(phrase.findall(r['text'])),len(r['text']),r['doc_id'],r['chunk_id']))
        out=[self._result(r,1.0,'exact') for r in exact[:top_k]]
        seen={(x['doc_id'],x['chunk_id']) for x in out}
        if len(out)>=top_k:return out

        terms=list(dict.fromkeys(_TOKEN.findall(query.casefold())))[:12]
        if terms:
            fts_query=' AND '.join('"'+t.replace('"','')+'"' for t in terms)
            try:
                sql="SELECT c.doc_id,c.chunk_id,c.text,d.path,d.version FROM chunks_fts f JOIN chunks c ON c.id=f.rowid JOIN documents d ON d.doc_id=f.doc_id WHERE chunks_fts MATCH ? AND d.status='active' ORDER BY bm25(chunks_fts) LIMIT ?"
                for rank,row in enumerate(self.db.execute(sql,(fts_query,limit)),1):
                    key=(row['doc_id'],row['chunk_id'])
                    if key not in seen:
                        out.append(self._result(row,0.95-min(rank,100)*0.001,'fts'));seen.add(key)
                        if len(out)>=top_k:return out
            except sqlite3.OperationalError:
                pass

        # Семантика дополняет, но никогда не вытесняет точные/лексические ответы.
        for item in super().search(query,top_k=limit,embedder=embedder):
            key=(item['doc_id'],item['chunk_id'])
            if key in seen:continue
            item['match_type']='semantic';out.append(item);seen.add(key)
            if len(out)>=top_k:break
        return out
