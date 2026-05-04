"""pytest configuration: добавляет server/ в sys.path, чтобы работали `from shared...`."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SERVER = ROOT / "server"
# Сначала server/ — именно там лежит пакет shared/
sys.path.insert(0, str(SERVER))
