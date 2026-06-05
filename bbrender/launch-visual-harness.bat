@echo off
setlocal

cd /d "%~dp0"

start "bbrender visual harness server" cmd /k "npm run serve"

timeout /t 2 /nobreak >nul
start "" "http://127.0.0.1:4173/tests/visual-harness.html"

endlocal
