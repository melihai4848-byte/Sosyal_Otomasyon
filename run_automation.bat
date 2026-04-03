@echo off
cd /d "%~dp0"
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_automation.ps1"
set "EXIT_CODE=%ERRORLEVEL%"

if "%EXIT_CODE%"=="130" (
    exit /b 0
)

if not "%EXIT_CODE%"=="0" (
    echo.
    echo Program hata kodu ile kapandi: %EXIT_CODE%
    pause
)

exit /b %EXIT_CODE%
