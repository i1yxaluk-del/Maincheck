"""
Лёгкое векторное хранилище для RAG без тяжёлых зависимостей.

Эмбеддинги берутся с Ollama (модель nomic-embed-text по умолчанию — 768 dim,
~100 МБ, отлично понимает русский). Если Ollama недоступен, хранилище работает
в «резервном» режиме на TF-IDF (скейл слов), чтобы ingest/list/remove оставались
рабочими даже без сети.

Формат на диске (self-contained, без chromadb/faiss):
    <store_dir>/
        meta.json          — {docs: {doc_id: {path, version, added_at, chunks}}}
        vectors.npz        — np.savez: vectors, norms, doc_ids, chunk_ids, texts

API:
    store = RagStore("data/rag_store")
    store.add_document(doc_id="fz-44/v2024-07", file_path=Path("fz44.docx"),
                       embedder=OllamaEmbedder())
    hits = store.search("согласно распоряжения", top_k=4, embedder=...)
    store.remove_document("fz-44/v2024-07")
    store.list_documents()
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Protocol

import hashlib

from . import garant_cleanup


_log = logging.getLogger("ai_suggester.rag")


# ── Интерфейс эмбеддера ───────────────────────────────────────
class Embedder(Protocol):
    dim: int
    name: str

    def embed(self, texts: List[str]) -> List[List[float]]: ...


# ── Реальный эмбеддер: Ollama ─────────────────────────────────
class OllamaEmbedder:
    """Использует Ollama /api/embeddings. Требует `ollama pull <model>`."""

    def __init__(self, model: str = "nomic-embed-text",
                 base_url: str = "http://localhost:11434",
                 timeout: float = 60.0):
        self.name = f"ollama:{model}"
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._dim: Optional[int] = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            # Ленивая инициализация на первом вызове embed
            self._dim = len(self.embed(["probe"])[0])
        return self._dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        import httpx
        out: List[List[float]] = []
        with httpx.Client(timeout=self.timeout) as client:
            for t in texts:
                r = client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": t},
                )
                r.raise_for_status()
                vec = r.json().get("embedding") or []
                if not vec:
                    raise RuntimeError(f"Ollama вернул пустой эмбеддинг для модели {self.model}")
                out.append(vec)
        return out


# ── Резервный эмбеддер: хеш-TF (без сети) ─────────────────────
class HashingEmbedder:
    """Простой hashing-trick TF-эмбеддинг (для офлайн-работы и тестов).

    Не заменяет качественный эмбеддинг, но позволяет индексировать/искать
    документы по лексическому пересечению, когда Ollama недоступен.
    """

    _TOKEN = re.compile(r"[\w\-а-яёА-ЯЁ]{2,}", re.UNICODE)

    def __init__(self, dim: int = 1024):
        self._dim = dim
        self.name = f"hashing:{dim}"

    @property
    def dim(self) -> int:
        return self._dim

    @staticmethod
    def _stable_hash(tok: str) -> int:
        # md5 первых 8 байт → детерминированный int (важно между запусками процесса,
        # т.к. built-in hash() в Python рандомизирован PYTHONHASHSEED).
        return int.from_bytes(hashlib.md5(tok.encode("utf-8")).digest()[:8], "big")

    def embed(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for t in texts:
            vec = [0.0] * self._dim
            for tok in self._TOKEN.findall(t.lower()):
                h = self._stable_hash(tok) % self._dim
                vec[h] += 1.0
            # L2-нормализация
            n = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / n for v in vec])
        return out


# ── Хранилище ─────────────────────────────────────────────────
@dataclass
class _Entry:
    vec: List[float]
    norm: float
    doc_id: str
    chunk_id: int
    text: str


@dataclass
class DocMeta:
    path: str
    version: str
    added_at: str
    chunks: int
    embedder: str
    text_len: int = 0
    stats: dict = field(default_factory=dict)


class RagStore:
    """Компактное векторное хранилище поверх JSON + .npz (numpy, если есть) или list-а."""

    def __init__(self, store_dir: str | os.PathLike[str]):
        self.dir = Path(store_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.dir / "meta.json"
        self.entries_path = self.dir / "entries.jsonl"

        self.docs: dict[str, DocMeta] = {}
        self.entries: List[_Entry] = []
        self._load()

    # --- persistence -------------------------------------------------
    def _load(self) -> None:
        if self.meta_path.exists():
            data = json.loads(self.meta_path.read_text(encoding="utf-8"))
            self.docs = {k: DocMeta(**v) for k, v in data.get("docs", {}).items()}
        if self.entries_path.exists():
            with self.entries_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    self.entries.append(_Entry(
                        vec=d["vec"], norm=d["norm"],
                        doc_id=d["doc_id"], chunk_id=d["chunk_id"], text=d["text"],
                    ))
        _log.info("RAG store загружен: %d документов, %d чанков", len(self.docs), len(self.entries))

    def _save(self) -> None:
        self.meta_path.write_text(
            json.dumps(
                {"docs": {k: v.__dict__ for k, v in self.docs.items()}},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        with self.entries_path.open("w", encoding="utf-8") as f:
            for e in self.entries:
                f.write(json.dumps({
                    "vec": e.vec, "norm": e.norm,
                    "doc_id": e.doc_id, "chunk_id": e.chunk_id, "text": e.text,
                }, ensure_ascii=False) + "\n")

    # --- API ---------------------------------------------------------
    def list_documents(self) -> List[dict]:
        return [{"doc_id": k, **v.__dict__} for k, v in sorted(self.docs.items())]

    def remove_document(self, doc_id: str) -> bool:
        if doc_id not in self.docs:
            return False
        self.docs.pop(doc_id)
        self.entries = [e for e in self.entries if e.doc_id != doc_id]
        self._save()
        _log.info("RAG: удалён документ %s", doc_id)
        return True

    def add_document(
        self,
        *,
        doc_id: str,
        file_path: Path,
        embedder: Embedder,
        version: str = "",
        chunk_chars: int = 1200,
        overlap: int = 150,
        replace: bool = True,
    ) -> DocMeta:
        """Очищает документ, режет на чанки, считает эмбеддинги и сохраняет."""
        if doc_id in self.docs and not replace:
            raise ValueError(f"Документ {doc_id} уже есть. Передайте replace=True для перезаписи.")

        cleaned, stats = garant_cleanup.extract_and_clean(Path(file_path))
        chunks = list(garant_cleanup.chunk_text(cleaned, chunk_chars=chunk_chars, overlap=overlap))
        if not chunks:
            raise ValueError(f"После очистки документ пуст: {file_path}")

        vectors = embedder.embed(chunks)
        if any(len(v) != embedder.dim for v in vectors):
            raise RuntimeError("Несогласованный размер эмбеддингов — проверьте модель эмбеддера")

        # Удаляем старую версию
        if doc_id in self.docs:
            self.entries = [e for e in self.entries if e.doc_id != doc_id]

        for i, (text, vec) in enumerate(zip(chunks, vectors)):
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            self.entries.append(_Entry(vec=vec, norm=norm, doc_id=doc_id, chunk_id=i, text=text))

        meta = DocMeta(
            path=str(file_path),
            version=version or time.strftime("%Y-%m-%dT%H:%M:%S"),
            added_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            chunks=len(chunks),
            embedder=embedder.name,
            text_len=len(cleaned),
            stats=stats.as_dict(),
        )
        self.docs[doc_id] = meta
        self._save()
        _log.info("RAG: добавлен %s (%d чанков, %d симв, embedder=%s)",
                  doc_id, len(chunks), len(cleaned), embedder.name)
        return meta

    def search(self, query: str, *, top_k: int = 4, embedder: Embedder) -> List[dict]:
        if not self.entries:
            return []
        qvec = embedder.embed([query])[0]
        qnorm = math.sqrt(sum(v * v for v in qvec)) or 1.0
        scored: List[tuple[float, _Entry]] = []
        for e in self.entries:
            if len(e.vec) != len(qvec):
                continue  # эмбеддер сменился
            dot = sum(a * b for a, b in zip(qvec, e.vec))
            scored.append((dot / (qnorm * e.norm), e))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"score": round(s, 4), "doc_id": e.doc_id, "chunk_id": e.chunk_id, "text": e.text}
            for s, e in scored[:top_k]
        ]
