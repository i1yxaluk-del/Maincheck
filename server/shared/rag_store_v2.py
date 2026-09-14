"""SQLite/FTS5 RAG: атомарное обновление, статусы и папочная синхронизация."""
from __future__ import annotations
import hashlib,json,math,os,re,sqlite3,time
from dataclasses import asdict,dataclass,field
from pathlib import Path
from typing import List,Protocol
from . import garant_cleanup
TOK=re.compile(r"[\w\-а-яёА-ЯЁ]{2,}",re.U); EXTS={'.txt','.md','.docx','.pdf','.html','.htm','.rtf'}
class Embedder(Protocol):
 name:str;dim:int
 def embed(self,texts:List[str])->List[List[float]]:...
class HashingEmbedder:
 def __init__(self,dim=1024):self.dim,self.name=dim,f'hashing:{dim}'
 def embed(self,texts):
  out=[]
  for text in texts:
   v=[0.0]*self.dim
   for t in TOK.findall(text.lower()):v[int.from_bytes(hashlib.md5(t.encode()).digest()[:8],'big')%self.dim]+=1
   n=math.sqrt(sum(x*x for x in v)) or 1;out.append([x/n for x in v])
  return out
class OllamaEmbedder:
 def __init__(self,model='nomic-embed-text',base_url='http://localhost:11434',timeout=60):self.model,self.base_url,self.timeout,self.name,self._dim=model,base_url.rstrip('/'),timeout,f'ollama:{model}',None
 @property
 def dim(self):return self._dim or len(self.embed(['probe'])[0])
 def embed(self,texts):
  import httpx
  out=[]
  with httpx.Client(timeout=self.timeout) as c:
   for text in texts:
    r=c.post(self.base_url+'/api/embeddings',json={'model':self.model,'prompt':text});r.raise_for_status();v=r.json().get('embedding') or []
    if not v:raise RuntimeError('Пустой эмбеддинг Ollama')
    self._dim=len(v);out.append(v)
  return out
@dataclass
class _Entry:vec:List[float];norm:float;doc_id:str;chunk_id:int;text:str
@dataclass
class DocMeta:
 path:str;version:str;added_at:str;chunks:int;embedder:str;text_len:int=0;stats:dict=field(default_factory=dict);updated_at:str='';file_hash:str='';content_hash:str='';status:str='active';operation:str=''
class RagStore:
 def __init__(self,store_dir):
  self.dir=Path(store_dir);self.dir.mkdir(parents=True,exist_ok=True);self.db_path=self.dir/'rag.sqlite3';self.db=sqlite3.connect(self.db_path,timeout=30);self.db.row_factory=sqlite3.Row
  self.db.executescript("""PRAGMA journal_mode=WAL;PRAGMA foreign_keys=ON;CREATE TABLE IF NOT EXISTS documents(doc_id TEXT PRIMARY KEY,path TEXT,version TEXT,added_at TEXT,updated_at TEXT,chunks INTEGER,embedder TEXT,text_len INTEGER,stats_json TEXT,file_hash TEXT,content_hash TEXT,status TEXT,chunk_chars INTEGER,overlap INTEGER);CREATE TABLE IF NOT EXISTS chunks(id INTEGER PRIMARY KEY,doc_id TEXT REFERENCES documents(doc_id) ON DELETE CASCADE,chunk_id INTEGER,text TEXT,vector_json TEXT,norm REAL,UNIQUE(doc_id,chunk_id));CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text,doc_id UNINDEXED,chunk_id UNINDEXED,tokenize='unicode61');""")
 def _meta(self,r,op=''):return DocMeta(r['path'],r['version'],r['added_at'],r['chunks'],r['embedder'],r['text_len'],json.loads(r['stats_json']),r['updated_at'],r['file_hash'],r['content_hash'],r['status'],op)
 @property
 def docs(self):return {r['doc_id']:self._meta(r) for r in self.db.execute('SELECT * FROM documents')}
 @property
 def entries(self):return [_Entry(json.loads(r['vector_json']),r['norm'],r['doc_id'],r['chunk_id'],r['text']) for r in self.db.execute("SELECT c.* FROM chunks c JOIN documents d USING(doc_id) WHERE d.status='active'")]
 def list_documents(self):return [{'doc_id':k,**asdict(v)} for k,v in sorted(self.docs.items())]
 def show_document(self,doc_id):
  r=self.db.execute('SELECT * FROM documents WHERE doc_id=?',(doc_id,)).fetchone()
  return None if not r else {'doc_id':doc_id,**asdict(self._meta(r)),'chunk_items':[dict(x) for x in self.db.execute('SELECT chunk_id,text FROM chunks WHERE doc_id=? ORDER BY chunk_id',(doc_id,))]}
 def add_document(self,*,doc_id,file_path,embedder,version='',chunk_chars=1200,overlap=150,replace=True):
  p=Path(file_path);fh=hashlib.sha256(p.read_bytes()).hexdigest();text,stats=garant_cleanup.extract_and_clean(p);ch=hashlib.sha256(text.encode()).hexdigest();old=self.db.execute('SELECT * FROM documents WHERE doc_id=?',(doc_id,)).fetchone();now=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
  if old and not replace:raise ValueError(f'Документ {doc_id} уже существует')
  if old and old['content_hash']==ch and old['embedder']==embedder.name and old['chunk_chars']==chunk_chars and old['overlap']==overlap:
   with self.db:self.db.execute("UPDATE documents SET path=?,version=?,updated_at=?,file_hash=?,status='active' WHERE doc_id=?",(str(p),version or old['version'],now,fh,doc_id))
   return self._meta(self.db.execute('SELECT * FROM documents WHERE doc_id=?',(doc_id,)).fetchone(),'unchanged')
  parts=list(garant_cleanup.chunk_text(text,chunk_chars=chunk_chars,overlap=overlap))
  if not parts:raise ValueError('Документ пуст после очистки')
  vectors=embedder.embed(parts);added=old['added_at'] if old else now
  with self.db:
   ids=[x[0] for x in self.db.execute('SELECT id FROM chunks WHERE doc_id=?',(doc_id,))]
   for i in ids:self.db.execute('DELETE FROM chunks_fts WHERE rowid=?',(i,))
   self.db.execute('DELETE FROM chunks WHERE doc_id=?',(doc_id,));self.db.execute('INSERT OR REPLACE INTO documents VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(doc_id,str(p),version or now,added,now,len(parts),embedder.name,len(text),json.dumps(stats.as_dict(),ensure_ascii=False),fh,ch,'active',chunk_chars,overlap))
   for i,(part,v) in enumerate(zip(parts,vectors)):
    n=math.sqrt(sum(x*x for x in v)) or 1;cur=self.db.execute('INSERT INTO chunks(doc_id,chunk_id,text,vector_json,norm) VALUES(?,?,?,?,?)',(doc_id,i,part,json.dumps(v,separators=(',',':')),n));self.db.execute('INSERT INTO chunks_fts(rowid,text,doc_id,chunk_id) VALUES(?,?,?,?)',(cur.lastrowid,part,doc_id,i))
  return self._meta(self.db.execute('SELECT * FROM documents WHERE doc_id=?',(doc_id,)).fetchone(),'updated' if old else 'added')
 def set_status(self,doc_id,status):
  if status not in {'active','disabled','missing'}:raise ValueError(status)
  with self.db:cur=self.db.execute('UPDATE documents SET status=?,updated_at=? WHERE doc_id=?',(status,time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),doc_id))
  return bool(cur.rowcount)
 def remove_document(self,doc_id):
  with self.db:
   for x in self.db.execute('SELECT id FROM chunks WHERE doc_id=?',(doc_id,)):self.db.execute('DELETE FROM chunks_fts WHERE rowid=?',(x[0],))
   cur=self.db.execute('DELETE FROM documents WHERE doc_id=?',(doc_id,))
  return bool(cur.rowcount)
 def scan_folder(self,folder):
  root=Path(folder).resolve();found={str(p.relative_to(root)).replace(os.sep,'/'):p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in EXTS};out=[]
  for i,p in sorted(found.items()):out.append({'doc_id':i,'path':str(p),'state':'unchanged' if i in self.docs and self.docs[i].file_hash==hashlib.sha256(p.read_bytes()).hexdigest() else ('changed' if i in self.docs else 'new')})
  for i,d in self.docs.items():
   if str(Path(d.path).resolve()).startswith(str(root)+os.sep) and i not in found:out.append({'doc_id':i,'path':d.path,'state':'missing'})
  return out
 def sync_folder(self,folder,*,embedder,chunk_chars=1200,overlap=150,prune=False):
  s={k:0 for k in ('added','updated','unchanged','missing','removed')};s['errors']=[]
  for x in self.scan_folder(folder):
   try:
    if x['state']=='missing':self.remove_document(x['doc_id']) if prune else self.set_status(x['doc_id'],'missing');s['removed' if prune else 'missing']+=1
    else:m=self.add_document(doc_id=x['doc_id'],file_path=Path(x['path']),embedder=embedder,chunk_chars=chunk_chars,overlap=overlap);s[m.operation]+=1
   except Exception as e:s['errors'].append({'doc_id':x['doc_id'],'error':str(e)})
  return s
 def search(self,query,*,top_k=4,embedder):
  q=embedder.embed([query])[0];qn=math.sqrt(sum(x*x for x in q)) or 1;rows=self.db.execute("SELECT c.*,d.path,d.version FROM chunks c JOIN documents d USING(doc_id) WHERE d.status='active' AND d.embedder=?",(embedder.name,));scores=[]
  for r in rows:
   v=json.loads(r['vector_json'])
   if len(v)==len(q):scores.append((sum(a*b for a,b in zip(q,v))/(qn*r['norm']),r))
  scores.sort(key=lambda x:x[0],reverse=True);return [{'score':round(s,4),'doc_id':r['doc_id'],'chunk_id':r['chunk_id'],'text':r['text'],'path':r['path'],'version':r['version']} for s,r in scores[:top_k]]
 def doctor(self):
  x=self.db.execute('PRAGMA integrity_check').fetchone()[0];return {'ok':x=='ok','integrity':x,'database':str(self.db_path),'documents':len(self.docs),'active_chunks':len(self.entries)}
