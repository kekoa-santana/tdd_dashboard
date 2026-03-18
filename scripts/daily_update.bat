@echo off
REM ──────────────────────────────────────────────────────────────
REM  TDD Dashboard — Daily Update Runner
REM  Called by Windows Task Scheduler.
REM
REM  Usage:
REM    daily_update.bat              — full update (projections + schedule)
REM    daily_update.bat --skip-schedule  — projections only
REM ──────────────────────────────────────────────────────────────

set PROJECT_DIR=C:\Users\kekoa\Documents\data_analytics\tdd-dashboard
set PYTHON=C:\Users\kekoa\AppData\Local\Programs\Python\Python311\python.exe
set LOG_DIR=%PROJECT_DIR%\logs
set LOG_FILE=%LOG_DIR%\update_%date:~-4,4%-%date:~-10,2%-%date:~-7,2%.log

REM Create logs directory if needed
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo [%date% %time%] Starting update... >> "%LOG_FILE%" 2>&1
"%PYTHON%" "%PROJECT_DIR%\scripts\update_in_season.py" %* >> "%LOG_FILE%" 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo [%date% %time%] Update FAILED with exit code %ERRORLEVEL% >> "%LOG_FILE%" 2>&1
) else (
    echo [%date% %time%] Update completed successfully >> "%LOG_FILE%" 2>&1
)
