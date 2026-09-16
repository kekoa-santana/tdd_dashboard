@echo off
setlocal EnableDelayedExpansion
REM ----------------------------------------------------------------
REM  TDD Dashboard -- Update Runner
REM  Called by Windows Task Scheduler.
REM
REM  Modes:
REM    daily_update.bat              -- morning run: ETL, precompute, projections,
REM                                     full game sims, publish
REM    daily_update.bat --skip-etl   -- morning run without the ETL step
REM    daily_update.bat --intraday   -- re-sim only games whose starters, umpire,
REM                                     or lineups changed; live standouts; publish
REM                                     ^(--schedule-only is accepted as an alias^)
REM
REM  Artifacts publish to the R2 bucket, not to git, so a data refresh never
REM  redeploys the Streamlit app. Requires requirements-pipeline.txt installed
REM  and R2_* credentials in .env.
REM
REM  Task Scheduler setup:
REM    1. TDD Full Daily Update:  6:00 AM             -> daily_update.bat
REM    2. TDD Intraday Refresh:   every 15 min, 9 AM to midnight
REM                                                   -> daily_update.bat --intraday
REM ----------------------------------------------------------------

set PROJECT_DIR=C:\Users\kekoa\Documents\data_analytics\tdd-dashboard
set PROFILES_DIR=C:\Users\kekoa\Documents\data_analytics\player_profiles
set ETL_DIR=C:\Users\kekoa\Documents\data_analytics\mlb_fantasy_ETL
set ETL_PYTHON=%ETL_DIR%\myenv\Scripts\python.exe
set PROFILES_PYTHON=%PROFILES_DIR%\myenv\Scripts\python.exe
set PYTHON=C:\Users\kekoa\AppData\Local\Programs\Python\Python311\python.exe
set LOG_DIR=%PROJECT_DIR%\logs
set LOG_FILE=%LOG_DIR%\update_%date:~-4,4%-%date:~-10,2%-%date:~-7,2%.log
REM Sim runner entry point. Its INFO log names the games each run re-simulates.
set PICKS=import sys, logging; logging.basicConfig(level=logging.WARNING); logging.getLogger('precompute.confident_picks').setLevel(logging.INFO); sys.path.insert(0, 'scripts/precompute'); sys.path.insert(0, 'scripts'); from confident_picks import run

REM Precompute groups that change with each day's games. Excludes the MCMC
REM model refit and preseason snapshots.
set DAILY_PRECOMPUTE_GROUPS=team,rankings,game_data,traditional,profiles

set IS_INTRADAY=0
echo %* | findstr /i "intraday schedule-only" >nul
if %ERRORLEVEL% EQU 0 set IS_INTRADAY=1

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM -- Single-instance lock: overlapping task firings skip instead of colliding --
REM   The lock records the owning cmd.exe PID. A run killed mid-flight (the
REM   scheduler time limit, a reboot, Ctrl-C) leaves a lock whose process is
REM   gone; the next run sees that and clears it. Waiting out a fixed timeout
REM   instead used to stall every refresh for hours.
set LOCK_FILE=%LOG_DIR%\update.lock
for /f %%i in ('powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter ('ProcessId=' + $PID)).ParentProcessId"') do set SELF_PID=%%i
if exist "%LOCK_FILE%" (
    for /f "usebackq delims=" %%s in (`powershell -NoProfile -Command "$p = (Get-Content '%LOCK_FILE%' -TotalCount 1).Trim(); $alive = $false; if ($p -match '^[0-9]+$') { $proc = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $p) -ErrorAction SilentlyContinue; if ($proc -and $proc.CommandLine -match 'daily_update') { $alive = $true } }; if ($alive -and (Get-Item '%LOCK_FILE%').LastWriteTime -gt (Get-Date).AddHours(-3)) { 'live' } else { 'stale' }"`) do set LOCK_STATE=%%s
    if "!LOCK_STATE!"=="live" (
        echo [%date% %time%] Skipping run, another update holds the lock >> "%LOG_FILE%" 2>&1
        goto skip_locked
    )
    echo [%date% %time%] Clearing stale lock left by a run that did not finish >> "%LOG_FILE%" 2>&1
    del "%LOCK_FILE%" >nul 2>&1
)
echo !SELF_PID!> "%LOCK_FILE%"

for /f %%i in ('powershell -NoProfile -Command "(Get-Date).AddDays(-1).ToString('yyyy-MM-dd')"') do set YESTERDAY=%%i
for /f %%i in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyy-MM-dd')"') do set TODAY=%%i

if "%IS_INTRADAY%"=="1" (
    echo [%date% %time%] Starting intraday refresh... >> "%LOG_FILE%" 2>&1
    call :intraday
) else (
    echo [%date% %time%] Starting morning update... >> "%LOG_FILE%" 2>&1
    call :morning %*
)
goto end


REM ================================================================
REM  Morning run
REM ================================================================
:morning

REM -- Step 1: ETL for yesterday, then the upcoming schedule --
echo %* | findstr /i "skip-etl" >nul
if %ERRORLEVEL% EQU 0 (
    echo [%date% %time%] Skipping ETL step >> "%LOG_FILE%" 2>&1
) else (
    echo [%date% %time%] Running ETL for %YESTERDAY%... >> "%LOG_FILE%" 2>&1
    cd /d "%ETL_DIR%"
    "%ETL_PYTHON%" "%ETL_DIR%\full_pipeline.py" --start-date %YESTERDAY% --end-date %YESTERDAY% >> "%LOG_FILE%" 2>&1
    if !ERRORLEVEL! NEQ 0 (
        echo [%date% %time%] ETL FAILED with exit code !ERRORLEVEL! -- continuing with existing data >> "%LOG_FILE%" 2>&1
    ) else (
        echo [%date% %time%] ETL completed successfully >> "%LOG_FILE%" 2>&1
    )
    "%ETL_PYTHON%" "%ETL_DIR%\ingestion\ingest_schedule_lookahead.py" --start-date %TODAY% --days 7 >> "%LOG_FILE%" 2>&1
    if !ERRORLEVEL! NEQ 0 echo [%date% %time%] Schedule lookahead FAILED -- continuing >> "%LOG_FILE%" 2>&1
)

REM -- Step 1b: Daily news feed + email digest --
echo [%date% %time%] Building daily news feed... >> "%LOG_FILE%" 2>&1
cd /d "%PROFILES_DIR%"
"%PROFILES_PYTHON%" -c "import sys; sys.path.insert(0, 'scripts/precompute'); sys.path.insert(0, 'scripts'); from news import run; run()" >> "%LOG_FILE%" 2>&1
if !ERRORLEVEL! NEQ 0 echo [%date% %time%] News feed FAILED -- continuing >> "%LOG_FILE%" 2>&1

REM -- Step 1c: Precompute the artifacts that move with each day's games --
REM   Rankings, team ELO/profiles, traditional stats, game-data priors, profiles.
echo [%date% %time%] Running precompute ^(%DAILY_PRECOMPUTE_GROUPS%^)... >> "%LOG_FILE%" 2>&1
cd /d "%PROFILES_DIR%"
"%PROFILES_PYTHON%" "%PROFILES_DIR%\scripts\precompute_dashboard_data.py" --include %DAILY_PRECOMPUTE_GROUPS% >> "%LOG_FILE%" 2>&1
if !ERRORLEVEL! NEQ 0 (
    echo [%date% %time%] Precompute FAILED -- continuing with existing artifacts >> "%LOG_FILE%" 2>&1
) else (
    echo [%date% %time%] Precompute completed successfully >> "%LOG_FILE%" 2>&1
)

REM -- Step 2: Projections + bookkeeping --
REM   Must follow the precompute: both steps write hitter_traditional.parquet
REM   and pitcher_traditional.parquet, the precompute with the last completed
REM   season and this step with the current one. The dashboard reads those as
REM   current-season stats, so the in-season write has to land last.
set DASH_ARGS=%*
if defined DASH_ARGS set DASH_ARGS=!DASH_ARGS:--skip-etl=!
echo [%date% %time%] Running dashboard update ^(projections + bookkeeping^)... >> "%LOG_FILE%" 2>&1
"%PYTHON%" "%PROJECT_DIR%\scripts\update_in_season.py" !DASH_ARGS! >> "%LOG_FILE%" 2>&1
if !ERRORLEVEL! NEQ 0 (
    echo [%date% %time%] Dashboard update FAILED with exit code !ERRORLEVEL! >> "%LOG_FILE%" 2>&1
) else (
    echo [%date% %time%] Dashboard update completed successfully >> "%LOG_FILE%" 2>&1
)

REM -- Step 3: Full game sims for today and tomorrow --
REM   Team-run scoring must cover the slate first, or the sim's run anchor
REM   silently falls back to the raw simulation.
echo [%date% %time%] Scoring team runs for scheduled games... >> "%LOG_FILE%" 2>&1
cd /d "%PROFILES_DIR%"
"%PROFILES_PYTHON%" "%PROFILES_DIR%\scripts\predict_team_runs_scheduled.py" >> "%LOG_FILE%" 2>&1
if !ERRORLEVEL! NEQ 0 echo [%date% %time%] Team run scoring FAILED -- sims will run without the run anchor >> "%LOG_FILE%" 2>&1

echo [%date% %time%] Running game sims ^(all games^)... >> "%LOG_FILE%" 2>&1
"%PROFILES_PYTHON%" -c "%PICKS%; run()" >> "%LOG_FILE%" 2>&1
if !ERRORLEVEL! NEQ 0 (
    echo [%date% %time%] Game sims FAILED >> "%LOG_FILE%" 2>&1
) else (
    echo [%date% %time%] Game sims completed successfully >> "%LOG_FILE%" 2>&1
)

call :post_sims
call :publish
exit /b 0


REM ================================================================
REM  Intraday refresh
REM ================================================================
:intraday

REM -- Step 1: Roster moves --
"%PYTHON%" "%PROJECT_DIR%\scripts\update_in_season.py" --schedule-only >> "%LOG_FILE%" 2>&1
if !ERRORLEVEL! NEQ 0 echo [%date% %time%] Roster move check FAILED -- continuing >> "%LOG_FILE%" 2>&1

REM -- Step 2: Re-sim only games whose inputs changed --
REM   Exit code 3 means nothing changed and no sim artifacts were touched.
cd /d "%PROFILES_DIR%"
"%PROFILES_PYTHON%" -c "%PICKS%; sys.exit(0 if run(incremental=True) else 3)" >> "%LOG_FILE%" 2>&1
set SIM_RESULT=!ERRORLEVEL!
if "!SIM_RESULT!"=="0" echo [%date% %time%] Changed games re-simulated >> "%LOG_FILE%" 2>&1
if "!SIM_RESULT!"=="3" echo [%date% %time%] No lineup, starter, or umpire changes >> "%LOG_FILE%" 2>&1
if not "!SIM_RESULT!"=="0" if not "!SIM_RESULT!"=="3" echo [%date% %time%] Game sims FAILED with exit code !SIM_RESULT! >> "%LOG_FILE%" 2>&1

REM -- Step 3: Live daily standouts + 14-day heat check from boxscores --
"%PROFILES_PYTHON%" -c "import sys; sys.path.insert(0, 'scripts/precompute'); sys.path.insert(0, 'scripts'); from rankings import run_live_standouts; run_live_standouts()" >> "%LOG_FILE%" 2>&1
if !ERRORLEVEL! NEQ 0 echo [%date% %time%] Live standouts FAILED >> "%LOG_FILE%" 2>&1

if "!SIM_RESULT!"=="0" call :post_sims
call :publish
exit /b 0


REM ================================================================
REM  Shared steps
REM ================================================================
:post_sims
echo [%date% %time%] Running post-sim bookkeeping... >> "%LOG_FILE%" 2>&1
"%PYTHON%" "%PROJECT_DIR%\scripts\update_in_season.py" --post-sims >> "%LOG_FILE%" 2>&1
if !ERRORLEVEL! NEQ 0 (
    echo [%date% %time%] Post-sim bookkeeping FAILED >> "%LOG_FILE%" 2>&1
) else (
    echo [%date% %time%] Post-sim bookkeeping completed >> "%LOG_FILE%" 2>&1
)
exit /b 0

:publish
REM   Only changed objects upload, and only the set the dashboard reads.
REM   Credentials come from .env ^(gitignored^).
cd /d "%PROJECT_DIR%"
"%PYTHON%" "%PROJECT_DIR%\scripts\publish_artifacts.py" --prune >> "%LOG_FILE%" 2>&1
if !ERRORLEVEL! NEQ 0 (
    echo [%date% %time%] Artifact publish FAILED with exit code !ERRORLEVEL! >> "%LOG_FILE%" 2>&1
) else (
    echo [%date% %time%] Artifacts published to R2 successfully >> "%LOG_FILE%" 2>&1
)
exit /b 0

:end
echo [%date% %time%] Update finished >> "%LOG_FILE%" 2>&1
del "%LOCK_FILE%" >nul 2>&1
:skip_locked
endlocal
