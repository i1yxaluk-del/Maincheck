from pathlib import Path
from shared.rag_store import HashingEmbedder,RagStore

def test_sync_skips_updates_and_marks_missing(tmp_path:Path):
 docs=tmp_path/'docs';docs.mkdir();p=docs/'order.txt';p.write_text('Приказ устанавливает порядок ведения учета.\n'*20,encoding='utf-8')
 s=RagStore(tmp_path/'store');e=HashingEmbedder(128)
 first=s.sync_folder(docs,embedder=e);assert first['added']==1
 second=s.sync_folder(docs,embedder=e);assert second['unchanged']==1
 p.write_text('Новая редакция устанавливает порядок ведения учета.\n'*20,encoding='utf-8')
 third=s.sync_folder(docs,embedder=e);assert third['updated']==1
 p.unlink();fourth=s.sync_folder(docs,embedder=e);assert fourth['missing']==1
 assert s.search('порядок учета',embedder=e)==[]
 assert s.doctor()['ok']

def test_enable_disable_and_atomic_replace(tmp_path:Path):
 p=tmp_path/'d.txt';p.write_text('служебно-боевая деятельность',encoding='utf-8');s=RagStore(tmp_path/'store');e=HashingEmbedder(64)
 s.add_document(doc_id='d',file_path=p,embedder=e);assert s.search('деятельность',embedder=e)
 assert s.set_status('d','disabled');assert not s.search('деятельность',embedder=e)
 assert s.set_status('d','active');assert s.search('деятельность',embedder=e)
