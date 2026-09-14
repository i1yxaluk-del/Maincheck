"""Локальное RAG-хранилище: SQLite как источник истины, FTS5 и векторы.

DOCX читается только при sync. Проверка текста обращается к единому индексу.
Смена документа атомарна: старая версия остаётся активной до готовности новой.
Redis необязателен и используется только как кеш результатов поиска.
"""
from __future__ import annotations

import hashlib, json, logging, math, os, re, sqlite3, threading, time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Protocol

from . import garant_cleanup

_log = logging.getLogger("ai_suggester.rag")
_ALLOWED = {".txt", ".md", ".docx", ".pdf", ".htm", ".html", ".rtf"}
_TOKEN = re.compile(r"[\w\-а-яёА-ЯЁ]{2,}", re.UNICODE)

class Embedder(Protocol):
    dim: int
    name: str
    def embed(self, texts: List[str]) -> List[List[float]]: ...

class OllamaEmbedder:
    def __init__(self, model="nomic-embed-text", base_url="http://localhost:11434", timeout=60.0):
        self.model, self.base_url, self.timeout = model, base_url.rstrip("/"), timeout
        self.name, self._dim = f"ollama:{model}", None
    @property
    def dim(self):
        if self._dim is None: self._dim = len(self.embed(["probe"])[0])
        return self._dim
    def embed(self, texts):
        import httpx
        out=[]
        with httpx.Client(timeout=self.timeout) as client:
            for text in texts:
                r=client.post(f"{self.base_url}/api/embeddings", json={"model":self.model,"prompt":text})
                r.raise_for_status(); vec=r.json().get("embedding") or []
                if not vec: raise RuntimeError(f"Ollama вернул пустой эмбеддинг: {self.model}")
                self._dim=len(vec); out.append(vec)
        return out

class HashingEmbedder:
    def __init__(self, dim=1024): self._dim, self.name = dim, f"hashing:{dim}"
    @property
    def dim(self): return self._dim
    def embed(self, texts):
        out=[]
        for text in texts:
            vec=[0.0]*self._dim
            for tok in _TOKEN.findall(text.lower()):
                i=int.from_bytes(hashlib.md5(tok.encode()).digest()[:8],"big")%self._dim; vec[i]+=1.0
            n=math.sqrt(sum(x*x for x in vec)) or 1.0; out.append([x/n for x in vec])
        return out

@dataclass
class _Entry:
    vec: List[float]; norm: float; doc_id: str; chunk_id: int; text: str

@dataclass
class DocMeta:
    path: str; version: str; added_at: str; chunks: int; embedder: str
    text_len: int=0; stats: dict=field(default_factory=dict); updated_at: str=""
    file_hash: str=""; content_hash: str=""; status: str="active"; operation: str=""

class RagStore:
    def __init__(self, store_dir):
        self.dir=Path(store_dir); self.dir.mkdir(parents=True, exist_ok=True)
        self.db_path=self.dir/"rag.sqlite3"; self._lock=threading.RLock()
        self.db=sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        self.db.row_factory=sqlite3.Row
        self.db.executescript("""
        PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA foreign_keys=ON; PRAGMA busy_timeout=30000;
        CREATE TABLE IF NOT EXISTS documents(
          doc_id TEXT PRIMARY KEY,path TEXT NOT NULL,version TEXT NOT NULL,added_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,chunks INTEGER NOT NULL,embedder TEXT NOT NULL,text_len INTEGER NOT NULL,
          stats_json TEXT NOT NULL,file_hash TEXT NOT NULL,content_hash TEXT NOT NULL,status TEXT NOT NULL DEFAULT 'active',
          chunk_chars INTEGER NOT NULL,overlap INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS chunks(
          id INTEGER PRIMARY KEY,doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
          chunk_id INTEGER NOT NULL,text TEXT NOT NULL,vector_json TEXT NOT NULL,norm REAL NOT NULL,
          UNIQUE(doc_id,chunk_id));
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text,doc_id UNINDEXED,chunk_id UNINDEXED,tokenize='unicode61');
        CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        INSERT OR IGNORE INTO settings VALUES('generation','0');
        """)
        self._migrate_legacy()

    def _migrate_legacy(self):
        meta, entries=self.dir/"meta.json", self.dir/"entries.jsonl"
        if self.db.execute("SELECT 1 FROM documents LIMIT 1").fetchone() or not (meta.exists() and entries.exists()): return
        try:
            docs=json.loads(meta.read_text(encoding="utf-8")).get("docs",{}); grouped={}
            for line in entries.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    e=json.loads(line); grouped.setdefault(e["doc_id"],[]).append(e)
            now=self._now()
            with self.db:
                for doc_id,d in docs.items():
                    es=grouped.get(doc_id,[])
                    self.db.execute("INSERT INTO documents VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(
                        doc_id,d.get("path",""),d.get("version",""),d.get("added_at",now),now,len(es),d.get("embedder","legacy"),
                        d.get("text_len",0),json.dumps(d.get("stats",{}),ensure_ascii=False),"","","active",1200,150))
                    for e in es: self._insert_chunk(doc_id,e["chunk_id"],e["text"],e["vec"],e.get("norm",1.0))
                self._bump()
            meta.rename(meta.with_suffix(".json.migrated")); entries.rename(entries.with_suffix(".jsonl.migrated"))
        except Exception as exc: _log.warning("Не удалось перенести прежний RAG: %s",exc)

    @staticmethod
    def _now(): return time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
    def _bump(self): self.db.execute("UPDATE settings SET value=CAST(value AS INTEGER)+1 WHERE key='generation'")
    def _generation(self): return self.db.execute("SELECT value FROM settings WHERE key='generation'").fetchone()[0]
    def _row_meta(self,r,operation=""):
        return DocMeta(r["path"],r["version"],r["added_at"],r["chunks"],r["embedder"],r["text_len"],json.loads(r["stats_json"]),r["updated_at"],r["file_hash"],r["content_hash"],r["status"],operation)
    @property
    def docs(self): return {r["doc_id"]:self._row_meta(r) for r in self.db.execute("SELECT * FROM documents")}
    @property
    def entries(self):
        rows=self.db.execute("SELECT c.* FROM chunks c JOIN documents d USING(doc_id) WHERE d.status='active' ORDER BY c.doc_id,c.chunk_id")
        return [_Entry(json.loads(r["vector_json"]),r["norm"],r["doc_id"],r["chunk_id"],r["text"]) for r in rows]
    def list_documents(self): return [{"doc_id":k,**asdict(v)} for k,v in sorted(self.docs.items())]
    def show_document(self,doc_id):
        r=self.db.execute("SELECT * FROM documents WHERE doc_id=?",(doc_id,)).fetchone()
        if not r:return None
        chunks=[dict(x) for x in self.db.execute("SELECT chunk_id,text FROM chunks WHERE doc_id=? ORDER BY chunk_id",(doc_id,))]
        return {"doc_id":doc_id,**asdict(self._row_meta(r)),"chunk_items":chunks}
    def _insert_chunk(self,doc_id,i,text,vec,norm):
        cur=self.db.execute("INSERT INTO chunks(doc_id,chunk_id,text,vector_json,norm) VALUES(?,?,?,?,?)",(doc_id,i,text,json.dumps(vec,separators=(",",":")),norm))
        self.db.execute("INSERT INTO chunks_fts(rowid,text,doc_id,chunk_id) VALUES(?,?,?,?)",(cur.lastrowid,text,doc_id,i))
    def add_document(self,*,doc_id,file_path,embedder,version="",chunk_chars=1200,overlap=150,replace=True):
        path=Path(file_path); raw=path.read_bytes(); file_hash=hashlib.sha256(raw).hexdigest()
        cleaned,stats=garant_cleanup.extract_and_clean(path)
        content_hash=hashlib.sha256(cleaned.encode()).hexdigest(); old=self.db.execute("SELECT * FROM documents WHERE doc_id=?",(doc_id,)).fetchone()
        if old and not replace: raise ValueError(f"Документ {doc_id} уже существует")
        now=self._now()
        if old and old["content_hash"]==content_hash and old["embedder"]==embedder.name and old["chunk_chars"]==chunk_chars and old["overlap"]==overlap:
            with self.db:
                self.db.execute("UPDATE documents SET path=?,version=?,updated_at=?,file_hash=?,status='active' WHERE doc_id=?",(str(path),version or old["version"],now,file_hash,doc_id)); self._bump()
            return self._row_meta(self.db.execute("SELECT * FROM documents WHERE doc_id=?",(doc_id,)).fetchone(),"unchanged")
        chunks=list(garant_cleanup.chunk_text(cleaned,chunk_chars=chunk_chars,overlap=overlap))
        if not chunks: raise ValueError(f"После очистки документ пуст: {path}")
        vectors=embedder.embed(chunks)
        if any(len(v)!=embedder.dim for v in vectors): raise RuntimeError("Размеры эмбеддингов различаются")
        added=old["added_at"] if old else now
        with self._lock,self.db:
            if old:
                ids=[r[0] for r in self.db.execute("SELECT id FROM chunks WHERE doc_id=?",(doc_id,))]
                for rowid in ids:self.db.execute("DELETE FROM chunks_fts WHERE rowid=?",(rowid,))
                self.db.execute("DELETE FROM chunks WHERE doc_id=?",(doc_id,))
            self.db.execute("INSERT OR REPLACE INTO documents VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(
                doc_id,str(path),version or now,added,now,len(chunks),embedder.name,len(cleaned),json.dumps(stats.as_dict(),ensure_ascii=False),
                file_hash,content_hash,"active",chunk_chars,overlap))
            for i,(text,vec) in enumerate(zip(chunks,vectors)):
                norm=math.sqrt(sum(x*x for x in vec)) or 1.0; self._insert_chunk(doc_id,i,text,vec,norm)
            self._bump()
        return self._row_meta(self.db.execute("SELECT * FROM documents WHERE doc_id=?",(doc_id,)).fetchone(),"updated" if old else "added")
    def set_status(self,doc_id,status):
        if status not in {"active","disabled","missing"}: raise ValueError(status)
        with self.db:
            cur=self.db.execute("UPDATE documents SET status=?,updated_at=? WHERE doc_id=?",(status,self._now(),doc_id));
            if cur.rowcount:self._bump()
        return bool(cur.rowcount)
    def remove_document(self,doc_id):
        with self.db:
            ids=[r[0] for r in self.db.execute("SELECT id FROM chunks WHERE doc_id=?",(doc_id,))]
            for rowid in ids:self.db.execute("DELETE FROM chunks_fts WHERE rowid=?",(rowid,))
            cur=self.db.execute("DELETE FROM documents WHERE doc_id=?",(doc_id,));
            if cur.rowcount:self._bump()
        return bool(cur.rowcount)
    def scan_folder(self,folder):
        root=Path(folder).resolve(); found={str(p.relative_to(root)).replace(os.sep,"/"):p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in _ALLOWED}; docs=self.docs
        result=[]
        for doc_id,p in sorted(found.items()):
            h=hashlib.sha256(p.read_bytes()).hexdigest(); result.append({"doc_id":doc_id,"path":str(p),"state":"unchanged" if doc_id in docs and docs[doc_id].file_hash==h else ("changed" if doc_id in docs else "new")})
        for doc_id,d in docs.items():
            try: inside=Path(d.path).resolve().is_relative_to(root)
            except AttributeError: inside=str(Path(d.path).resolve()).startswith(str(root)+os.sep)
            if inside and doc_id not in found: result.append({"doc_id":doc_id,"path":d.path,"state":"missing"})
        return result
    def sync_folder(self,folder,*,embedder,chunk_chars=1200,overlap=150,prune=False):
        root=Path(folder).resolve(); plan=self.scan_folder(root); summary={"added":0,"updated":0,"unchanged":0,"missing":0,"removed":0,"errors":[]}
        for item in plan:
            try:
                if item["state"]=="missing":
                    if prune:self.remove_document(item["doc_id"]);summary["removed"]+=1
                    else:self.set_status(item["doc_id"],"missing");summary["missing"]+=1
                else:
                    m=self.add_document(doc_id=item["doc_id"],file_path=Path(item["path"]),embedder=embedder,chunk_chars=chunk_chars,overlap=overlap)
                    summary[m.operation]+=1
            except Exception as exc: summary["errors"].append({"doc_id":item["doc_id"],"error":str(exc)})
        return summary
    def search(self,query,*,top_k=4,embedder):
        if not query.strip():return []
        q=embedder.embed([query])[0]; qn=math.sqrt(sum(x*x for x in q)) or 1.0; scored=[]
        rows=self.db.execute("SELECT c.doc_id,c.chunk_id,c.text,c.vector_json,c.norm,d.path,d.version FROM chunks c JOIN documents d USING(doc_id) WHERE d.status='active' AND d.embedder=?",(embedder.name,))
        for r in rows:
            v=json.loads(r["vector_json"])
            if len(v)==len(q): scored.append((sum(a*b for a,b in zip(q,v))/(qn*r["norm"]),r))
        scored.sort(key=lambda x:x[0],reverse=True); ranks={}
        for rank,(score,r) in enumerate(scored[:100],1): ranks[(r["doc_id"],r["chunk_id"])]=[0.75/(60+rank),r,score]
        terms=list(dict.fromkeys(_TOKEN.findall(query.lower())))[:12]
        if terms:
            try:
                fquery=" OR ".join('"'+t.replace('"','')+'"' for t in terms)
                for rank,r in enumerate(self.db.execute("SELECT f.doc_id,f.chunk_id,c.text,d.path,d.version FROM chunks_fts f JOIN chunks c ON c.id=f.rowid JOIN documents d ON d.doc_id=f.doc_id WHERE chunks_fts MATCH ? AND d.status='active' LIMIT 100",(fquery,)),1):
                    key=(r["doc_id"],r["chunk_id"]); ranks.setdefault(key,[0.0,r,0.0])[0]+=0.25/(60+rank)
            except sqlite3.OperationalError: pass
        out=[]
        for _,r,vscore in sorted(ranks.values(),key=lambda x:x[0],reverse=True)[:top_k]: out.append({"score":round(vscore,4),"doc_id":r["doc_id"],"chunk_id":r["chunk_id"],"text":r["text"],"path":r["path"],"version":r["version"]})
        return out
    def doctor(self):
        integ=self.db.execute("PRAGMA integrity_check").fetchone()[0]
        return {"ok":integ=="ok","integrity":integ,"database":str(self.db_path),"generation":int(self._generation()),"documents":len(self.docs),"active_documents":self.db.execute("SELECT count(*) FROM documents WHERE status='active'").fetchone()[0],"active_chunks":self.db.execute("SELECT count(*) FROM chunks c JOIN documents d USING(doc_id) WHERE d.status='active'").fetchone()[0]}
