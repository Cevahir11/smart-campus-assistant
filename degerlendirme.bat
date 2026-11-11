@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo === Baseline sonuclari (Project Card alan 8) ===
".venv\Scripts\python.exe" evaluation\midproject.py
echo.
echo === Sinif bazinda precision / recall / F1 ===
".venv\Scripts\python.exe" evaluation\per_class_report.py
echo.
echo === Hata kategorileri (alan 9) ===
".venv\Scripts\python.exe" evaluationrror_analysis.py
echo.
echo === Tum asamalar: siniflandirma, bilgi cikarimi, ozetleme, soru-cevap ===
".venv\Scripts\python.exe" evaluationvaluate.py
echo.
echo === Testler ===
".venv\Scripts\python.exe" -m pytest -q
pause
