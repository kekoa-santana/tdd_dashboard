@echo off
setlocal EnableDelayedExpansion
REM ----------------------------------------------------------------
REM  TDD Dashboard -- Update Runner
REM  Called by Windows Task Scheduler.
REM
REM  Modes:
REM    daily_update.bat                 -- full update (ETL + projections + sims + push)
REM    daily_update.bat --skip-etl      -- skip ETL, projections + sims + push
REM    daily_update.bat --schedule-only -- roster moves + sims + push (single run)
REM
REM  Task Scheduler setup:
REM    1. Full daily:   6:00 AM  -> daily_update.bat
REM    2. Game window:  every 10 min (8 AM-4 PM) -> daily_update.bat --schedule-only
REM                     Task Scheduler handles repetition; each invocation runs once.
REM ----------------------------------------------------------------

set PROJECT_DIR=C:\Users\kekoa\Documents\data_analytics\tdd-dashboard
set DEPLOY_DIR=C:\Users\kekoa\Documents\data_analytics\tdd-dashboard-deploy
set PROFILES_DIR=C:\Users\kekoa\Documents\data_analytics\player_profiles
set ETL_DIR=C:\Users\kekoa\Documents\data_analytics\mlb_fantasy_ETL
set ETL_PYTHON=%ETL_DIR%\myenv\Scripts\python.exe
set PROFILES_PYTHON=%PROFILES_DIR%\myenv\Scripts\python.exe
set PYTHON=C:\Users\kekoa\AppData\Local\Programs\Python\Python311\python.exe
set LOG_DIR=%PROJECT_DIR%\logs
set LOG_FILE=%LOG_DIR%\update_%date:~-4,4%-%date:~-10,2%-%date:~-7,2%.log
set IS_SCHEDULE_ONLY=0
echo %* | findstr /i "schedule-only" >nul
if %ERRORLEVEL% EQU 0 set IS_SCHEDULE_ONLY=1

REM Create logs directory if needed
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM -- Single-instance lock: overlapping task firings skip instead of colliding --
REM   A lock older than 2 hours is treated as stale and removed.
set LOCK_FILE=%LOG_DIR%\update.lock
if exist "%LOCK_FILE%" powershell -NoProfile -Command "if ((Get-Item '%LOCK_FILE%').LastWriteTime -lt (Get-Date).AddHours(-2)) { Remove-Item -Force '%LOCK_FILE%' }" >nul 2>&1
if exist "%LOCK_FILE%" (
    echo [%date% %time%] Skipping run, another update holds the lock >> "%LOG_FILE%" 2>&1
    goto skip_locked
)
echo locked > "%LOCK_FILE%"

if "%IS_SCHEDULE_ONLY%"=="1" (
    echo [%date% %time%] Starting schedule-only refresh... >> "%LOG_FILE%" 2>&1
) else (
    echo [%date% %time%] Starting full update... >> "%LOG_FILE%" 2>&1
)

REM -- Compute yesterday's date (for ETL) --
for /f %%i in ('powershell -NoProfile -Command "(Get-Date).AddDays(-1).ToString('yyyy-MM-dd')"') do set YESTERDAY=%%i

call :run_once %*
goto end

:run_once
REM -- Step 1: ETL -- skip for --schedule-only or --skip-etl --
echo %* | findstr /i "schedule-only skip-etl" >nul
if %ERRORLEVEL% EQU 0 (
    echo [%date% %time%] Skipping ETL step >> "%LOG_FILE%" 2>&1
) else (
    echo [%date% %time%] Running ETL for %YESTERDAY%... >> "%LOG_FILE%" 2>&1
    cd /d "%ETL_DIR%"
    "%ETL_PYTHON%" "%ETL_DIR%\full_pipeline.py" --start-date %YESTERDAY% --end-date %YESTERDAY% >> "%LOG_FILE%" 2>&1
    if !ERRORLEVEL! NEQ 0 (
        echo [%date% %time%] ETL FAILED with exit code !ERRORLEVEL! >> "%LOG_FILE%" 2>&1
        echo [%date% %time%] Continuing with dashboard update using existing data... >> "%LOG_FILE%" 2>&1
    ) else (
        echo [%date% %time%] ETL completed successfully >> "%LOG_FILE%" 2>&1
    )
)

REM -- Step 1b: Precompute -- SKIPPED during season (preseason data is static) --
REM   Rankings and team profiles update weekly via --weekly flag on update_in_season.py.
REM   To re-run preseason precompute manually:
REM     cd /d "%PROFILES_DIR%" && "%PROFILES_PYTHON%" scripts\precompute_dashboard_data.py --include team,rankings,game_data,traditional,profiles,game_sim,health

REM -- Step 1c: Daily news feed + email digest --
REM   External RSS (MLB.com/MiLB.com) + DB-generated stories -> news_feed.parquet
REM   plus the morning email digest (needs GMAIL_* in player_profiles\.env).
if "%IS_SCHEDULE_ONLY%"=="0" (
    echo [%date% %time%] Building daily news feed... >> "%LOG_FILE%" 2>&1
    cd /d "%PROFILES_DIR%"
    "%PROFILES_PYTHON%" -c "import sys; sys.path.insert(0, 'scripts/precompute'); sys.path.insert(0, 'scripts'); from news import run; run()" >> "%LOG_FILE%" 2>&1
    if !ERRORLEVEL! NEQ 0 (
        echo [%date% %time%] News feed FAILED -- continuing >> "%LOG_FILE%" 2>&1
    ) else (
        echo [%date% %time%] News feed generated successfully >> "%LOG_FILE%" 2>&1
    )
)

REM -- Step 2: Dashboard update (projections + bookkeeping, NO sims) --
if "%IS_SCHEDULE_ONLY%"=="1" (
    echo [%date% %time%] Running dashboard schedule-only ^(roster moves + odds^)... >> "%LOG_FILE%" 2>&1
    "%PYTHON%" "%PROJECT_DIR%\scripts\update_in_season.py" --schedule-only >> "%LOG_FILE%" 2>&1
) else (
    set DASH_ARGS=%*
    if defined DASH_ARGS (
        set DASH_ARGS=!DASH_ARGS:--skip-etl=!
    )
    echo [%date% %time%] Running dashboard update ^(projections + bookkeeping^)... >> "%LOG_FILE%" 2>&1
    "%PYTHON%" "%PROJECT_DIR%\scripts\update_in_season.py" !DASH_ARGS! >> "%LOG_FILE%" 2>&1
)

if %ERRORLEVEL% NEQ 0 (
    echo [%date% %time%] Dashboard update FAILED with exit code %ERRORLEVEL% >> "%LOG_FILE%" 2>&1
    echo [%date% %time%] Continuing to sims... >> "%LOG_FILE%" 2>&1
) else (
    echo [%date% %time%] Dashboard update completed successfully >> "%LOG_FILE%" 2>&1
)

REM -- Step 3: Run game sims via confident_picks (sole sim runner) --
REM   Fetches schedule + lineups from MLB API, runs pitcher + batter sims,
REM   produces: game_props.parquet, todays_games.parquet,
REM   todays_lineups.parquet, todays_batter_sims.parquet, game_predictions.parquet.
echo [%date% %time%] Running game sims ^(confident_picks^)... >> "%LOG_FILE%" 2>&1
cd /d "%PROFILES_DIR%"
"%PROFILES_PYTHON%" -c "import sys; sys.path.insert(0, 'scripts/precompute'); sys.path.insert(0, 'scripts'); from confident_picks import run; run()" >> "%LOG_FILE%" 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo [%date% %time%] Game sims FAILED >> "%LOG_FILE%" 2>&1
) else (
    echo [%date% %time%] Game sims completed successfully >> "%LOG_FILE%" 2>&1
)

REM -- Step 3b: Live daily standouts + 14-day heat check --
REM   Fetches completed-game boxscores from MLB API and rebuilds
REM   daily standout + weekly form parquets with today's results.
if "%IS_SCHEDULE_ONLY%"=="1" (
    echo [%date% %time%] Generating live standouts... >> "%LOG_FILE%" 2>&1
    cd /d "%PROFILES_DIR%"
    "%PROFILES_PYTHON%" -c "import sys; sys.path.insert(0, 'scripts/precompute'); sys.path.insert(0, 'scripts'); from rankings import run_live_standouts; run_live_standouts()" >> "%LOG_FILE%" 2>&1
    if !ERRORLEVEL! NEQ 0 (
        echo [%date% %time%] Live standouts FAILED >> "%LOG_FILE%" 2>&1
    ) else (
        echo [%date% %time%] Live standouts generated successfully >> "%LOG_FILE%" 2>&1
    )
)

REM -- Step 3c: Post-sim bookkeeping, game predictions reshape + odds + metadata --
echo [%date% %time%] Running post-sim bookkeeping... >> "%LOG_FILE%" 2>&1
"%PYTHON%" "%PROJECT_DIR%\scripts\update_in_season.py" --post-sims >> "%LOG_FILE%" 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo [%date% %time%] Post-sim bookkeeping FAILED >> "%LOG_FILE%" 2>&1
) else (
    echo [%date% %time%] Post-sim bookkeeping completed >> "%LOG_FILE%" 2>&1
)

REM -- Step 4: Publish data to origin/main via deploy worktree --
REM   The dev checkout can sit on any branch; publishing happens from a
REM   detached worktree pinned to origin/main so branches never collide.
REM   Streamlit Cloud serves main, so this is the deploy step.
echo [%date% %time%] Publishing data to origin/main... >> "%LOG_FILE%" 2>&1
cd /d "%PROJECT_DIR%"

if not exist "%DEPLOY_DIR%\.git" (
    git worktree add --detach "%DEPLOY_DIR%" origin/main >> "%LOG_FILE%" 2>&1
)

git -C "%DEPLOY_DIR%" fetch origin main >> "%LOG_FILE%" 2>&1
git -C "%DEPLOY_DIR%" reset --hard origin/main >> "%LOG_FILE%" 2>&1

REM Copy the published artifact set: top-level parquet/json/npz/pkl + snapshots.
REM   /PURGE removes deploy-side files that no longer exist locally, so deleted
REM   artifacts actually leave main instead of being restored by the reset above.
REM   /XD keeps purge away from the subdirectories: history is gitignored and
REM   never published, snapshots is mirrored separately on the next line.
robocopy "%PROJECT_DIR%\data\dashboard" "%DEPLOY_DIR%\data\dashboard" *.parquet *.json *.npz *.pkl /PURGE /XD history snapshots /NJH /NJS /NDL /NP >> "%LOG_FILE%" 2>&1
robocopy "%PROJECT_DIR%\data\dashboard\snapshots" "%DEPLOY_DIR%\data\dashboard\snapshots" /MIR /NJH /NJS /NDL /NP >> "%LOG_FILE%" 2>&1

REM robocopy uses 0-7 for success (1 = files copied); only 8+ is a real failure.
if !ERRORLEVEL! GEQ 8 echo [%date% %time%] Artifact copy FAILED with code !ERRORLEVEL! >> "%LOG_FILE%" 2>&1

REM -A so removals are staged, not just modifications and additions
git -C "%DEPLOY_DIR%" add -A data/dashboard >> "%LOG_FILE%" 2>&1
git -C "%DEPLOY_DIR%" diff --cached --quiet
if %ERRORLEVEL% NEQ 0 (
    git -C "%DEPLOY_DIR%" commit -m "data update" >> "%LOG_FILE%" 2>&1
    git -C "%DEPLOY_DIR%" push origin HEAD:main >> "%LOG_FILE%" 2>&1
    if !ERRORLEVEL! NEQ 0 (
        echo [%date% %time%] Git push FAILED >> "%LOG_FILE%" 2>&1
    ) else (
        echo [%date% %time%] Data published to origin/main successfully >> "%LOG_FILE%" 2>&1
    )
) else (
    echo [%date% %time%] No data changes to publish >> "%LOG_FILE%" 2>&1
)
exit /b 0

:end
echo [%date% %time%] Update finished >> "%LOG_FILE%" 2>&1
del "%LOCK_FILE%" >nul 2>&1
:skip_locked
endlocal
