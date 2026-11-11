@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
rem Bilgisayar Muhendisligi duyurularini GTU sitesinden yeniden ceker (en yeni 40)
".venv\Scripts\python.exe" scripts\collect_gtu.py --dept ceng --limit 40
pause
