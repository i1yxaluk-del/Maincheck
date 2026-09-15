"""Публичный RAG API для local/cloud и фонового индексатора."""
from .rag_store_v2 import *  # noqa: F401,F403
from .rag_ollama import OllamaEmbedder  # noqa: F401,E402
from .rag_chunking import safe_chunk_text  # noqa: E402
from .rag_office_extract import read_office_legacy  # noqa: E402
from . import rag_store_v2 as _rag_store_v2  # noqa: E402

_rag_store_v2.garant_cleanup.chunk_text=safe_chunk_text
_rag_store_v2.garant_cleanup._EXTRACTORS['.doc']=read_office_legacy
_rag_store_v2.garant_cleanup._EXTRACTORS['.rtf']=read_office_legacy
_rag_store_v2.EXTS.update({'.doc','.rtf'})

from .rag_hybrid import HybridRagStore  # noqa: E402
RagStore=HybridRagStore
