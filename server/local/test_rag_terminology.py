from pathlib import Path
from shared.rag_store import HashingEmbedder,RagStore
from rag_terminology import RagTerminologyStage

def test_wrong_abbreviation_is_corrected_from_canonical_alias(tmp_path:Path):
 p=tmp_path/'names.txt';p.write_text('Федеральное казенное учреждение (ФКУ «НИЦ «Охрана» Росгвардии).',encoding='utf-8')
 store=RagStore(tmp_path/'state');store.add_document(doc_id='names',file_path=p,embedder=HashingEmbedder(64),chunk_chars=300,overlap=20)
 stage=RagTerminologyStage(str(tmp_path/'state'))
 text='Сотрудников, находящихся в распоряжении ФГКУ\n«НИЦ «Охрана» Росгвардии не имеется.'
 found=stage.candidates(text)
 assert len(found)==1
 assert found[0].before=='ФГКУ' and found[0].after=='ФКУ'
 assert found[0].category=='rule-rag-terminology'

def test_correct_alias_is_untouched(tmp_path:Path):
 p=tmp_path/'names.txt';p.write_text('(ФКУ «НИЦ «Охрана» Росгвардии)',encoding='utf-8');store=RagStore(tmp_path/'state');store.add_document(doc_id='names',file_path=p,embedder=HashingEmbedder(64))
 assert RagTerminologyStage(str(tmp_path/'state')).candidates('ФКУ «НИЦ «Охрана» Росгвардии')==[]
