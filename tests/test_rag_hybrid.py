from pathlib import Path
from shared.rag_store import HashingEmbedder,RagStore

def test_exact_abbreviation_precedes_semantic_noise(tmp_path:Path):
 docs=tmp_path/'docs';docs.mkdir()
 (docs/'names.txt').write_text('ФГКУ — федеральное государственное казенное учреждение. '*20,encoding='utf-8')
 (docs/'noise.txt').write_text('Государственные учреждения культуры и музеи Российской Федерации. '*20,encoding='utf-8')
 store=RagStore(tmp_path/'state');emb=HashingEmbedder(64);store.sync_folder(docs,embedder=emb,chunk_chars=300,overlap=30)
 hits=store.search('ФГКУ',top_k=4,embedder=emb)
 assert hits and hits[0]['doc_id']=='names.txt'
 assert hits[0]['match_type']=='exact' and hits[0]['score']==1.0
 assert 'ФГКУ' in hits[0]['text']

def test_missing_document_is_not_returned_exactly(tmp_path:Path):
 docs=tmp_path/'docs';docs.mkdir();p=docs/'a.txt';p.write_text('ФГКУ специального назначения. '*20,encoding='utf-8')
 store=RagStore(tmp_path/'state');emb=HashingEmbedder(64);store.sync_folder(docs,embedder=emb);p.unlink();store.sync_folder(docs,embedder=emb)
 assert store.search('ФГКУ',embedder=emb)==[]
