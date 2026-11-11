@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
rem TUM bolumlerin duyurularini GTU sitesinden ceker (bolum basina en yeni 40, saniyede 1 istek, ~15-20 dk)
".venv\Scripts\python.exe" scripts\collect_gtu.py --all --limit 40
pause
