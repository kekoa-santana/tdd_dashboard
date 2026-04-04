@echo off
REM ──────────────────────────────────────────────────────────────
REM  TDD Dashboard — Create Windows Task Scheduler Tasks
REM  Run this ONCE as Administrator to set up automated updates.
REM
REM  Creates two tasks:
REM    1. "TDD Morning Update" — 7:00 AM PST (10 AM ET) daily
REM       Full pipeline: ETL (yesterday's games) + projections + bookkeeping.
REM       Runs ASAP if PC was off at 7 AM. 1-hour timeout.
REM    2. "TDD Hourly Update"  — every hour 8 AM–4 PM PST
REM       Schedule/lineup/sims refresh only (no ETL, no projections).
REM       Also runs ASAP on wake if missed.
REM
REM  To remove: run  remove_scheduled_tasks.bat
REM ──────────────────────────────────────────────────────────────

set TASK_DIR=C:\Users\kekoa\Documents\data_analytics\tdd-dashboard\scripts\tasks
if not exist "%TASK_DIR%" mkdir "%TASK_DIR%"

echo.
echo === Setting up TDD Dashboard scheduled tasks ===
echo.

REM ── Generate XML for Morning Update ──
(
echo ^<?xml version="1.0" encoding="UTF-16"?^>
echo ^<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task"^>
echo   ^<RegistrationInfo^>
echo     ^<Description^>TDD Dashboard — full pipeline: ETL + projections + bookkeeping (daily at 7 AM PST^)^</Description^>
echo   ^</RegistrationInfo^>
echo   ^<Triggers^>
echo     ^<CalendarTrigger^>
echo       ^<StartBoundary^>2026-03-18T07:00:00^</StartBoundary^>
echo       ^<Enabled^>true^</Enabled^>
echo       ^<ScheduleByDay^>
echo         ^<DaysInterval^>1^</DaysInterval^>
echo       ^</ScheduleByDay^>
echo     ^</CalendarTrigger^>
echo   ^</Triggers^>
echo   ^<Settings^>
echo     ^<MultipleInstancesPolicy^>IgnoreNew^</MultipleInstancesPolicy^>
echo     ^<DisallowStartIfOnBatteries^>false^</DisallowStartIfOnBatteries^>
echo     ^<StopIfGoingOnBatteries^>false^</StopIfGoingOnBatteries^>
echo     ^<AllowHardTerminate^>true^</AllowHardTerminate^>
echo     ^<StartWhenAvailable^>true^</StartWhenAvailable^>
echo     ^<RunOnlyIfNetworkAvailable^>true^</RunOnlyIfNetworkAvailable^>
echo     ^<AllowStartOnDemand^>true^</AllowStartOnDemand^>
echo     ^<Enabled^>true^</Enabled^>
echo     ^<ExecutionTimeLimit^>PT1H^</ExecutionTimeLimit^>
echo   ^</Settings^>
echo   ^<Actions Context="Author"^>
echo     ^<Exec^>
echo       ^<Command^>C:\Users\kekoa\Documents\data_analytics\tdd-dashboard\scripts\daily_update.bat^</Command^>
echo     ^</Exec^>
echo   ^</Actions^>
echo ^</Task^>
) > "%TASK_DIR%\morning_update.xml"

schtasks /create /tn "TDD Morning Update" /xml "%TASK_DIR%\morning_update.xml" /f

if %ERRORLEVEL% EQU 0 (
    echo [OK] Created "TDD Morning Update" - 7:00 AM PST daily, 1hr timeout, catches up on wake
) else (
    echo [FAIL] Could not create morning task. Run as Administrator.
)

REM ── Generate XML for Hourly Update ──
(
echo ^<?xml version="1.0" encoding="UTF-16"?^>
echo ^<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task"^>
echo   ^<RegistrationInfo^>
echo     ^<Description^>TDD Dashboard — 30-min schedule/lineup/sims refresh (8 AM-4 PM PST, skips if no changes^)^</Description^>
echo   ^</RegistrationInfo^>
echo   ^<Triggers^>
echo     ^<CalendarTrigger^>
echo       ^<StartBoundary^>2026-03-18T08:00:00^</StartBoundary^>
echo       ^<Enabled^>true^</Enabled^>
echo       ^<Repetition^>
echo         ^<Interval^>PT30M^</Interval^>
echo         ^<Duration^>PT8H^</Duration^>
echo         ^<StopAtDurationEnd^>false^</StopAtDurationEnd^>
echo       ^</Repetition^>
echo       ^<ScheduleByDay^>
echo         ^<DaysInterval^>1^</DaysInterval^>
echo       ^</ScheduleByDay^>
echo     ^</CalendarTrigger^>
echo   ^</Triggers^>
echo   ^<Settings^>
echo     ^<MultipleInstancesPolicy^>IgnoreNew^</MultipleInstancesPolicy^>
echo     ^<DisallowStartIfOnBatteries^>false^</DisallowStartIfOnBatteries^>
echo     ^<StopIfGoingOnBatteries^>false^</StopIfGoingOnBatteries^>
echo     ^<AllowHardTerminate^>true^</AllowHardTerminate^>
echo     ^<StartWhenAvailable^>true^</StartWhenAvailable^>
echo     ^<RunOnlyIfNetworkAvailable^>true^</RunOnlyIfNetworkAvailable^>
echo     ^<AllowStartOnDemand^>true^</AllowStartOnDemand^>
echo     ^<Enabled^>true^</Enabled^>
echo     ^<ExecutionTimeLimit^>PT15M^</ExecutionTimeLimit^>
echo   ^</Settings^>
echo   ^<Actions Context="Author"^>
echo     ^<Exec^>
echo       ^<Command^>C:\Users\kekoa\Documents\data_analytics\tdd-dashboard\scripts\daily_update.bat^</Command^>
echo       ^<Arguments^>--schedule-only^</Arguments^>
echo     ^</Exec^>
echo   ^</Actions^>
echo ^</Task^>
) > "%TASK_DIR%\hourly_update.xml"

schtasks /create /tn "TDD Hourly Update" /xml "%TASK_DIR%\hourly_update.xml" /f

if %ERRORLEVEL% EQU 0 (
    echo [OK] Created "TDD Hourly Update" - every 30 min 8 AM-4 PM PST, skips if no changes, catches up on wake
) else (
    echo [FAIL] Could not create hourly task. Run as Administrator.
)

echo.
echo === Done! ===
echo.
echo Both tasks have "Start when available" enabled:
echo   - If your PC is off at 7 AM, the morning update runs as soon as you turn it on
echo   - Hourly updates then continue automatically through 4 PM PST
echo.
echo Verify with:  schtasks /query /tn "TDD*" /v /fo list
echo.
pause
