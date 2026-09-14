"""Совместимый фасад RAG v2: SQLite/FTS5 и атомарная синхронизация."""
from .rag_store_v2 import *  # noqa: F401,F403
from .rag_ollama import OllamaEmbedder  # noqa: F401,E402
from .rag_chunking import safe_chunk_text  # noqa: E402
from .rag_legacy_doc import read_legacy_doc  # noqa: E402
from . import rag_store_v2 as _rag_store_v2  # noqa: E402

# Патчи применяются к общему модулю извлечения: local и cloud используют один API.
_rag_store_v2.garant_cleanup.chunk_text = safe_chunk_text
_rag_store_v2.garant_cleanup._EXTRACTORS[".doc"] = read_legacy_doc
_rag_store_v2.EXTS.add(".doc")
