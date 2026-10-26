@echo off
cd /d "%~dp0"
echo Building dist ...
make build
if errorlevel 1 (
  echo.
  echo build FAILED.
  pause
  exit /b 1
)
echo.
echo build complete.
echo   dist\packed\sb_bbrender.html     (all-in-one file, works from anywhere)
echo   dist\separate\sb_bbrender.html   (needs bbrender.css and bbrender.js beside it)
echo   dist\cdn\sb_bbrender.html        (hashed assets\ folder for CDN hosting)

pause
