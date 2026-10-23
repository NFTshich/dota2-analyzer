@echo off
title Dota2 Analyzer
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" goto ERR_NOVENV

echo ============================================================
echo   Dota2 Analyzer - match predictor
echo ============================================================
echo.
echo IMPORTANT: KEEP THIS WINDOW OPEN while using the analyzer.
echo To stop, press Ctrl+C in this window.
echo.
echo Address:  http://localhost:8501
echo Auto-refresh enabled (once per hour).
echo ============================================================
echo.

REM -- open browser after a short delay --
start "" cmd /c "timeout /t 8 >nul & start http://localhost:8501"

REM -- run streamlit from the venv --
".venv\Scripts\python.exe" -m streamlit run app.py --server.headless=true

echo.
echo Finished.
pause
exit /b 0

:ERR_NOVENV
echo [ERROR] Virtual environment .venv not found.
echo.
echo To fix, open Command Prompt IN this folder and run:
echo   python -m venv .venv
echo   .venv\Scripts\activate
echo   pip install -r requirements.txt
echo.
echo Then run this file again.
pause
exit /b 1