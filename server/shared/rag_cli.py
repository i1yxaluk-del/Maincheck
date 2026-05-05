"""
CLI для управления RAG-хранилищем.

Примеры:
    # Добавить документ
    python -m shared.rag_cli add ./docs/fz44.docx --doc-id fz-44 --version 2024-07

    # Просмотреть все документы
    python -m shared.rag_cli list

    # Удалить (например, документ отменён)
    python -m shared.rag_cli remove fz-44

    # Заменить на новую редакцию (replace по умолчанию включён)
    python -m shared.rag_cli add ./docs/fz44_new.docx --doc-id fz-44 --version 2025-03

    # Массовая загрузка из папки (watch-скан)
    python -m shared.rag_cli ingest-folder ./data/docs

    # Проверить индекс запросом
    python -m shared.rag_cli search "согласно распоряжения"

    # Предпросмотр очистки документа без индексации (dry-run) — удобно для
    # инспекции свежего файла Гарант/Консультант перед массовой загрузкой
    python -m shared.rag_cli preview ./docs/fz44.docx --head 2000

    # Сводная статистика по индексу (общий размер, эмбеддер и т.д.)
    python -m shared.rag_cli stats
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from collections import Counter

from . import garant_cleanup, rag_store as rs


def _make_embedder(args) -> rs.Embedder:
    if args.embedder == "hashing":
        return rs.HashingEmbedder(dim=args.hashing_dim)
    return rs.OllamaEmbedder(
        model=args.embed_model,
        base_url=args.ollama_url,
    )


def cmd_add(args) -> int:
    store = rs.RagStore(args.store_dir)
    embedder = _make_embedder(args)
    path = Path(args.file)
    if not path.exists():
        print(f"Файл не найден: {path}", file=sys.stderr)
        return 2
    doc_id = args.doc_id or path.stem
    meta = store.add_document(
        doc_id=doc_id,
        file_path=path,
        embedder=embedder,
        version=args.version or "",
        chunk_chars=args.chunk_chars,
        overlap=args.overlap,
        replace=True,
    )
    print(json.dumps({"doc_id": doc_id, **meta.__dict__}, ensure_ascii=False, indent=2))
    return 0


def cmd_list(args) -> int:
    store = rs.RagStore(args.store_dir)
    docs = store.list_documents()
    if args.json:
        print(json.dumps(docs, ensure_ascii=False, indent=2))
    else:
        if not docs:
            print("Хранилище пусто.")
            return 0
        fmt = "{:<30} {:<15} {:>6} {:>8}  {}"
        print(fmt.format("doc_id", "version", "chunks", "text_len", "embedder"))
        print("─" * 90)
        for d in docs:
            print(fmt.format(d["doc_id"], d["version"], d["chunks"], d["text_len"], d["embedder"]))
    return 0


def cmd_remove(args) -> int:
    store = rs.RagStore(args.store_dir)
    ok = store.remove_document(args.doc_id)
    print("Удалён." if ok else "Документ не найден.")
    return 0 if ok else 1


def cmd_search(args) -> int:
    store = rs.RagStore(args.store_dir)
    embedder = _make_embedder(args)
    hits = store.search(args.query, top_k=args.top_k, embedder=embedder)
    if args.json:
        print(json.dumps(hits, ensure_ascii=False, indent=2))
    else:
        for h in hits:
            print(f"[{h['score']:.3f}] {h['doc_id']}#chunk{h['chunk_id']}")
            print("    " + h["text"][:300].replace("\n", " ") + ("…" if len(h["text"]) > 300 else ""))
            print()
    return 0


def cmd_preview(args) -> int:
    """Прогнать очистку и чанкинг документа БЕЗ индексации.

    Полезно перед массовой загрузкой ведомственных документов из Гарант/
    Консультант — позволяет увидеть, что останется после удаления служебной
    разметки, как файл порежется на чанки, и нет ли подозрительных потерь.
    """
    path = Path(args.file)
    if not path.exists():
        print(f"Файл не найден: {path}", file=sys.stderr)
        return 2
    cleaned, stats = garant_cleanup.extract_and_clean(path)
    chunks = list(garant_cleanup.chunk_text(
        cleaned, chunk_chars=args.chunk_chars, overlap=args.overlap,
    ))
    summary = {
        "file": str(path),
        "cleanup": stats.as_dict(),
        "text_len": len(cleaned),
        "chunks": len(chunks),
        "chunk_chars_target": args.chunk_chars,
        "chunk_overlap": args.overlap,
        "first_chunk_chars": len(chunks[0]) if chunks else 0,
        "last_chunk_chars": len(chunks[-1]) if chunks else 0,
    }
    if args.json:
        print(json.dumps(
            {**summary, "head": cleaned[: args.head] if args.head else cleaned},
            ensure_ascii=False, indent=2,
        ))
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if args.head:
            print("\n── Превью очищенного текста ──")
            print(cleaned[: args.head])
            if len(cleaned) > args.head:
                print(f"… (ещё {len(cleaned) - args.head} символов опущено)")
    return 0


def cmd_stats(args) -> int:
    """Сводка по индексу без обращения к Ollama."""
    store = rs.RagStore(args.store_dir)
    if not store.docs:
        print("Хранилище пусто.")
        return 0
    embedders = Counter(d.embedder for d in store.docs.values())
    versions = Counter(d.version or "(без версии)" for d in store.docs.values())
    summary = {
        "store_dir": str(store.dir),
        "documents": len(store.docs),
        "chunks": len(store.entries),
        "vec_dim": len(store.entries[0].vec) if store.entries else 0,
        "embedders": dict(embedders),
        "versions_top5": dict(versions.most_common(5)),
        "text_chars_total": sum(d.text_len for d in store.docs.values()),
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_ingest_folder(args) -> int:
    store = rs.RagStore(args.store_dir)
    embedder = _make_embedder(args)
    folder = Path(args.folder)
    exts = {".txt", ".md", ".docx", ".pdf", ".htm", ".html", ".rtf"}
    added, skipped = 0, 0
    for f in sorted(folder.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in exts:
            continue
        doc_id = str(f.relative_to(folder)).replace(os.sep, "/")
        try:
            store.add_document(
                doc_id=doc_id,
                file_path=f,
                embedder=embedder,
                chunk_chars=args.chunk_chars,
                overlap=args.overlap,
                replace=True,
            )
            print(f"  + {doc_id}")
            added += 1
        except Exception as e:
            print(f"  ! {doc_id}: {e}", file=sys.stderr)
            skipped += 1
    print(f"\nЗагружено: {added}, пропущено: {skipped}")
    return 0 if skipped == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rag_cli",
        description="Управление RAG-хранилищем AI LibreOffice Suggester",
    )
    p.add_argument("--store-dir", default=os.getenv("RAG_STORE_DIR", "data/rag_store"),
                   help="Папка со структурами RAG (по умолчанию data/rag_store)")
    p.add_argument("--embedder", choices=("ollama", "hashing"), default="ollama",
                   help="Какой эмбеддер использовать (ollama — рекомендуется)")
    p.add_argument("--embed-model", default=os.getenv("RAG_EMBED_MODEL", "nomic-embed-text"))
    p.add_argument("--ollama-url", default=os.getenv("OLLAMA_URL", "http://localhost:11434"))
    p.add_argument("--hashing-dim", type=int, default=1024)

    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="Добавить документ")
    a.add_argument("file")
    a.add_argument("--doc-id", help="Идентификатор (по умолчанию — имя файла)")
    a.add_argument("--version", default="")
    a.add_argument("--chunk-chars", type=int, default=1200)
    a.add_argument("--overlap", type=int, default=150)
    a.set_defaults(func=cmd_add)

    l = sub.add_parser("list", help="Список документов")
    l.add_argument("--json", action="store_true")
    l.set_defaults(func=cmd_list)

    r = sub.add_parser("remove", help="Удалить документ")
    r.add_argument("doc_id")
    r.set_defaults(func=cmd_remove)

    s = sub.add_parser("search", help="Найти фрагменты")
    s.add_argument("query")
    s.add_argument("--top-k", type=int, default=4)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_search)

    f = sub.add_parser("ingest-folder", help="Массовая загрузка папки")
    f.add_argument("folder")
    f.add_argument("--chunk-chars", type=int, default=1200)
    f.add_argument("--overlap", type=int, default=150)
    f.set_defaults(func=cmd_ingest_folder)

    pv = sub.add_parser(
        "preview",
        help="Dry-run очистки документа без индексации (для проверки Гарант/Консультант)",
    )
    pv.add_argument("file")
    pv.add_argument("--chunk-chars", type=int, default=1200)
    pv.add_argument("--overlap", type=int, default=150)
    pv.add_argument("--head", type=int, default=2000,
                    help="Сколько символов очищенного текста показать (0 — всё)")
    pv.add_argument("--json", action="store_true")
    pv.set_defaults(func=cmd_preview)

    st = sub.add_parser("stats", help="Сводка по индексу")
    st.add_argument("--json", action="store_true")
    st.set_defaults(func=cmd_stats)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
