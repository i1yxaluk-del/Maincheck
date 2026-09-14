"""Совместимая точка входа CLI RAG v2."""
import sys

# Astra/Linux может вернуть имена файлов с surrogateescape, если имя записано
# не в UTF-8. Показываем такие байты как \udcXX вместо аварийного завершения CLI.
for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if _reconfigure is not None:
        _reconfigure(encoding="utf-8", errors="backslashreplace")

from .rag_cli_v2 import *  # noqa: E402,F401,F403

if __name__ == "__main__":
    raise SystemExit(main())
