@echo off
chcp 65001 > nul
echo ===================================
echo  AI Suggester — Local Model Server
echo ===================================
cd /d "%~dp0"
pip install -r requirements.txt -q
echo.
echo Сервер: http://localhost:8000
echo Статус Ollama: http://localhost:8000/health
echo.
uvicorn main:app --host 0.0.0.0 --port 8000
pause
