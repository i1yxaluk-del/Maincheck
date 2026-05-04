"""
Общий пакет AI LibreOffice Suggester.
Подключается в main.py каждого сервера через:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

Содержимое:
    logging_setup — логи с ротацией (TimedRotatingFileHandler + консоль)
    audit          — SQLite-аудит запросов (retention по LOG_RETENTION_DAYS)
"""
