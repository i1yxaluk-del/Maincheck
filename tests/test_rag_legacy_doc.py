from shared import rag_store

def test_legacy_doc_is_registered_for_scan():
    assert '.doc' in rag_store._rag_store_v2.EXTS
    assert '.doc' in rag_store._rag_store_v2.garant_cleanup._EXTRACTORS
