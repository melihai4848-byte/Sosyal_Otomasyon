@echo off
setlocal

cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" run_automation.py
) else (
    where py >nul 2>nul
    if %ERRORLEVEL%==0 (
        py run_automation.py
    ) else (
        where python >nul 2>nul
        if %ERRORLEVEL%==0 (
            python run_automation.py
        ) else (
            echo [Launcher] Python bulunamadi. Lutfen Python 3.10+ kur.
            pause
            exit /b 1
        )
    )
)
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo Program hata kodu ile kapandi: %EXIT_CODE%
    pause
)

exit /b %EXIT_CODE%
