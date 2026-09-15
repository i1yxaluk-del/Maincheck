from pathlib import Path
from shared.rag_store import HashingEmbedder,RagStore

class Counting(HashingEmbedder):
 def __init__(self):super().__init__(64);self.calls=0
 def embed(self,texts):self.calls+=1;return super().embed(texts)

def test_unchanged_file_is_not_reembedded(tmp_path:Path):
 docs=tmp_path/'docs';docs.mkdir();p=docs/'a.txt';p.write_text('Порядок ведения учета. '*50,encoding='utf-8')
 store=RagStore(tmp_path/'state');emb=Counting()
 assert store.sync_folder(docs,embedder=emb)['added']==1
 calls=emb.calls
 result=store.sync_folder(docs,embedder=emb)
 assert result['unchanged']==1 and emb.calls==calls

def test_changed_file_replaces_and_missing_is_inactive(tmp_path:Path):
 docs=tmp_path/'docs';docs.mkdir();p=docs/'a.txt';p.write_text('Старая редакция. '*50,encoding='utf-8');store=RagStore(tmp_path/'state');emb=Counting();store.sync_folder(docs,embedder=emb)
 p.write_text('Новая редакция. '*50,encoding='utf-8');assert store.sync_folder(docs,embedder=emb)['updated']==1
 p.unlink();assert store.sync_folder(docs,embedder=emb)['missing']==1;assert not store.search('редакция',embedder=emb)
