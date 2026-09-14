"""CLI управления документами: scan/sync/list/show/search/status/doctor."""
from __future__ import annotations
import argparse,json,os,sys
from collections import Counter
from pathlib import Path
from . import garant_cleanup
from . import rag_store as rs

def _emb(a): return rs.HashingEmbedder(a.hashing_dim) if a.embedder=="hashing" else rs.OllamaEmbedder(a.embed_model,a.ollama_url)
def _store(a): return rs.RagStore(a.store_dir)
def _dump(x): print(json.dumps(x,ensure_ascii=False,indent=2))
def cmd_add(a):
 p=Path(a.file)
 if not p.exists():print(f"Файл не найден: {p}",file=sys.stderr);return 2
 m=_store(a).add_document(doc_id=a.doc_id or p.stem,file_path=p,embedder=_emb(a),version=a.version,chunk_chars=a.chunk_chars,overlap=a.overlap);_dump({"doc_id":a.doc_id or p.stem,**m.__dict__});return 0
def cmd_list(a):
 docs=_store(a).list_documents()
 if a.json:_dump(docs);return 0
 if not docs:print("Хранилище пусто.");return 0
 print(f"{'doc_id':<36} {'status':<10} {'version':<20} {'chunks':>7}")
 for d in docs:print(f"{d['doc_id']:<36} {d['status']:<10} {d['version']:<20} {d['chunks']:>7}")
 return 0
def cmd_show(a):
 d=_store(a).show_document(a.doc_id)
 if not d:print("Документ не найден.",file=sys.stderr);return 1
 if not a.chunks:d.pop("chunk_items",None)
 _dump(d);return 0
def cmd_remove(a): print("Удалён." if _store(a).remove_document(a.doc_id) else "Документ не найден.");return 0
def cmd_status(a): print("Готово." if _store(a).set_status(a.doc_id,a.status) else "Документ не найден.");return 0
def cmd_search(a):
 hits=_store(a).search(a.query,top_k=a.top_k,embedder=_emb(a))
 if a.json:_dump(hits)
 else:
  for h in hits:print(f"[{h['score']:.3f}] {h['doc_id']}#{h['chunk_id']} ({h['version']})\n    {h['text'][:400].replace(chr(10),' ')}\n")
 return 0
def cmd_scan(a):
 plan=_store(a).scan_folder(a.folder);_dump(plan) if a.json else [print(f"{x['state']:<10} {x['doc_id']}") for x in plan];return 0
def cmd_sync(a):
 s=_store(a).sync_folder(a.folder,embedder=_emb(a),chunk_chars=a.chunk_chars,overlap=a.overlap,prune=a.prune);_dump(s);return 1 if s['errors'] else 0
def cmd_preview(a):
 p=Path(a.file)
 if not p.exists():print(f"Файл не найден: {p}",file=sys.stderr);return 2
 text,stats=garant_cleanup.extract_and_clean(p);chunks=list(garant_cleanup.chunk_text(text,chunk_chars=a.chunk_chars,overlap=a.overlap));payload={"file":str(p),"cleanup":stats.as_dict(),"text_len":len(text),"chunks":len(chunks),"chunk_chars_target":a.chunk_chars,"chunk_overlap":a.overlap,"first_chunk_chars":len(chunks[0]) if chunks else 0,"last_chunk_chars":len(chunks[-1]) if chunks else 0}
 if a.json:_dump({**payload,"head":text[:a.head] if a.head else text})
 else:_dump(payload);print("\n── Превью очищенного текста ──\n"+text[:a.head]) if a.head else None
 return 0
def cmd_stats(a):
 s=_store(a);docs=s.list_documents()
 if not docs:print("Хранилище пусто.");return 0
 payload={"store_dir":str(s.dir),"documents":len(docs),"chunks":len(s.entries),"vec_dim":len(s.entries[0].vec) if s.entries else 0,"embedders":dict(Counter(d['embedder'] for d in docs)),"versions_top5":dict(Counter(d['version'] for d in docs).most_common(5)),"text_chars_total":sum(d['text_len'] for d in docs)};_dump(payload);return 0
def cmd_doctor(a):_dump(_store(a).doctor());return 0

def build_parser():
 p=argparse.ArgumentParser(prog="ragctl",description="Управление локальной памятью RAG")
 p.add_argument("--store-dir",default=os.getenv("RAG_STORE_DIR","data/rag_store"));p.add_argument("--embedder",choices=("ollama","hashing"),default=os.getenv("RAG_EMBEDDER","ollama"));p.add_argument("--embed-model",default=os.getenv("RAG_EMBED_MODEL","nomic-embed-text"));p.add_argument("--ollama-url",default=os.getenv("OLLAMA_URL","http://localhost:11434"));p.add_argument("--hashing-dim",type=int,default=1024)
 sub=p.add_subparsers(dest="cmd",required=True)
 def chunks(x):x.add_argument("--chunk-chars",type=int,default=1200);x.add_argument("--overlap",type=int,default=150)
 a=sub.add_parser("add");a.add_argument("file");a.add_argument("--doc-id");a.add_argument("--version",default="");chunks(a);a.set_defaults(func=cmd_add)
 l=sub.add_parser("list");l.add_argument("--json",action="store_true");l.set_defaults(func=cmd_list)
 sh=sub.add_parser("show");sh.add_argument("doc_id");sh.add_argument("--chunks",action="store_true");sh.set_defaults(func=cmd_show)
 for name,status in (("enable","active"),("disable","disabled")):
  x=sub.add_parser(name);x.add_argument("doc_id");x.set_defaults(func=cmd_status,status=status)
 r=sub.add_parser("remove");r.add_argument("doc_id");r.set_defaults(func=cmd_remove)
 se=sub.add_parser("search");se.add_argument("query");se.add_argument("--top-k",type=int,default=4);se.add_argument("--json",action="store_true");se.set_defaults(func=cmd_search)
 sc=sub.add_parser("scan");sc.add_argument("folder");sc.add_argument("--json",action="store_true");sc.set_defaults(func=cmd_scan)
 sy=sub.add_parser("sync",aliases=["ingest-folder"]);sy.add_argument("folder");sy.add_argument("--prune",action="store_true");chunks(sy);sy.set_defaults(func=cmd_sync)
 pv=sub.add_parser("preview");pv.add_argument("file");pv.add_argument("--head",type=int,default=2000);pv.add_argument("--json",action="store_true");chunks(pv);pv.set_defaults(func=cmd_preview)
 st=sub.add_parser("stats");st.add_argument("--json",action="store_true");st.set_defaults(func=cmd_stats)
 d=sub.add_parser("doctor");d.set_defaults(func=cmd_doctor)
 return p
def main(argv=None):a=build_parser().parse_args(argv);return a.func(a)
