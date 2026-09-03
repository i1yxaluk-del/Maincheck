"""
Общий пакет AI LibreOffice Suggester.
Подключается в main.py каждого сервера через:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

Содержимое:
    logging_setup — логи с ротацией (TimedRotatingFileHandler + консоль)
    audit          — SQLite-аудит запросов (retention по LOG_RETENTION_DAYS)
"""

# v2.3: install a conservative adjective+noun agreement override before
# local/cloud servers import shared.morph_detector. It only fixes the
# preposition-led blind spot and is enabled by default with an emergency
# environment rollback flag.
from . import morph_detector_override as _morph_detector_override  # noqa: F401,E402
