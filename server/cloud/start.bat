@echo off
chcp 65001 > nul
echo ===================================
echo  AI LibreOffice Suggester Server
echo ===================================
cd /d "%~dp0"
echo Проверка зависимостей...
pip install -r requirements.txt -q
echo.
echo Сервер: http://localhost:8000
echo Проверка API: http://localhost:8000/test_api
echo Остановить: Ctrl+C
echo.
uvicorn main:app --host 0.0.0.0 --port 8000
pause
