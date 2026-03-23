@echo off
REM ──────────────────────────────────────────────────────────────
REM  TDD Dashboard — Daily Update Runner
REM  Called by Windows Task Scheduler.
REM
REM  Full pipeline:
REM    1. ETL — ingest yesterday's games into PostgreSQL
REM    2. Projections — conjugate update + game sims (delegates to player_profiles)
REM    3. Dashboard bookkeeping — roster export, snapshots, manifest
REM
REM  Usage:
REM    daily_update.bat                 — full update (ETL + projections + schedule)
REM    daily_update.bat --skip-etl      — skip ETL, projections + schedule only
REM    daily_update.bat --schedule-only — hourly mode (no ETL, no projections)
REM ──────────────────────────────────────────────────────────────

set PROJECT_DIR=C:\Users\kekoa\Documents\data_analytics\tdd-dashboard
set ETL_DIR=C:\Users\kekoa\Documents\data_analytics\mlb_fantasy_ETL
set ETL_PYTHON=%ETL_DIR%\myenv\Scripts\python.exe
set PYTHON=C:\Users\kekoa\AppData\Local\Programs\Python\Python311\python.exe
set LOG_DIR=%PROJECT_DIR%\logs
set LOG_FILE=%LOG_DIR%\update_%date:~-4,4%-%date:~-10,2%-%date:~-7,2%.log

REM Create logs directory if needed
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo [%date% %time%] Starting daily update... >> "%LOG_FILE%" 2>&1

REM ── Compute yesterday's date (for ETL) ──
REM   PowerShell one-liner to get yesterday in YYYY-MM-DD format
for /f %%i in ('powershell -NoProfile -Command "(Get-Date).AddDays(-1).ToString('yyyy-MM-dd')"') do set YESTERDAY=%%i

REM ── Step 1: ETL — skip for --schedule-only or --skip-etl ──
echo %* | findstr /i "schedule-only skip-etl" >nul
if %ERRORLEVEL% EQU 0 (
    echo [%date% %time%] Skipping ETL step >> "%LOG_FILE%" 2>&1
) else (
    echo [%date% %time%] Running ETL for %YESTERDAY%... >> "%LOG_FILE%" 2>&1
    "%ETL_PYTHON%" "%ETL_DIR%\full_pipeline.py" --start-date %YESTERDAY% --end-date %YESTERDAY% >> "%LOG_FILE%" 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo [%date% %time%] ETL FAILED with exit code %ERRORLEVEL% >> "%LOG_FILE%" 2>&1
        echo [%date% %time%] Continuing with dashboard update using existing data... >> "%LOG_FILE%" 2>&1
    ) else (
        echo [%date% %time%] ETL completed successfully >> "%LOG_FILE%" 2>&1
    )
)

REM ── Step 2-3: Dashboard update (projections + bookkeeping) ──
REM   Strip --skip-etl from args before passing to update_in_season.py
set DASH_ARGS=%*
if defined DASH_ARGS (
    set DASH_ARGS=%DASH_ARGS:--skip-etl=%
)
echo [%date% %time%] Running dashboard update... >> "%LOG_FILE%" 2>&1
"%PYTHON%" "%PROJECT_DIR%\scripts\update_in_season.py" %DASH_ARGS% >> "%LOG_FILE%" 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo [%date% %time%] Dashboard update FAILED with exit code %ERRORLEVEL% >> "%LOG_FILE%" 2>&1
) else (
    echo [%date% %time%] Daily update completed successfully >> "%LOG_FILE%" 2>&1
)
