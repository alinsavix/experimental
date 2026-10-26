@echo off
cd /d "%~dp0"
echo Building and installing ...
make install
if errorlevel 1 (
  echo.
  echo install FAILED.
  pause
  exit /b 1
)
echo.
echo install complete.

pause
