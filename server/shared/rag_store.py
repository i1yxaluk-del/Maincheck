"""Совместимый фасад RAG v2: SQLite/FTS5, версии и атомарная синхронизация."""
from .rag_store_v2 import *  # noqa: F401,F403
# Современный пакетный клиент перекрывает прежнюю совместимую реализацию.
from .rag_ollama import OllamaEmbedder  # noqa: F401,E402
