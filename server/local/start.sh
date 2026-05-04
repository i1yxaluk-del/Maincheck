#!/bin/bash
cd "$(dirname "$0")"
pip install -r requirements.txt -q
echo "Сервер: http://localhost:8000"
echo "Статус Ollama: http://localhost:8000/health"
uvicorn main:app --host 0.0.0.0 --port 8000
