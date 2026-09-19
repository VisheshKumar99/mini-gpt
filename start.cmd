@echo off
REM Starts the backend (FastAPI/uvicorn) and frontend (Vite) in parallel on Windows.
REM Each runs in its own console window. Close the windows to stop them.
REM
REM Usage:
REM   start.cmd

set "ROOT_DIR=%~dp0"

echo Starting backend (uvicorn) on http://localhost:8000 ...
start "mini-gpt backend" cmd /k "cd /d "%ROOT_DIR%" && uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000"

echo Starting frontend (vite) on http://localhost:5173 ...
start "mini-gpt frontend" cmd /k "cd /d "%ROOT_DIR%dashboard" && npm run dev"

echo.
echo Backend and frontend launched in separate windows.
