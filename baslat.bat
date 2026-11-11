@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo Web arayuzu baslatiliyor: http://127.0.0.1:5000  (kapatmak icin bu pencereyi kapatin)
start "" http://127.0.0.1:5000
".venv\Scripts\python.exe" run.py
pause
