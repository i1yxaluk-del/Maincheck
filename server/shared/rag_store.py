"""Совместимый фасад RAG v2: SQLite/FTS5 и атомарная синхронизация."""
from .rag_store_v2 import *  # noqa: F401,F403
from .rag_ollama import OllamaEmbedder  # noqa: F401,E402
from .rag_chunking import safe_chunk_text  # noqa: E402
from . import rag_store_v2 as _rag_store_v2  # noqa: E402

# v2 обращается к функции через модуль, поэтому замена действует и для старого
# публичного API без дублирования хранилища и без изменений cloud-сервера.
_rag_store_v2.garant_cleanup.chunk_text = safe_chunk_text
