# ──────────────────────────────────────────────────────────────
#  TDD Dashboard — Task Scheduler Setup
#  Run as Administrator:  powershell -ExecutionPolicy Bypass -File setup_scheduler.ps1
# ──────────────────────────────────────────────────────────────

$BatPath = "C:\Users\kekoa\Documents\data_analytics\tdd-dashboard\scripts\daily_update.bat"
$TaskFolder = "\TDD"

# ── Task 1: Full daily update (6 AM) ─────────────────────────
$FullName = "TDD Full Daily Update"
$FullTrigger = New-ScheduledTaskTrigger -Daily -At "6:00AM"
$FullAction  = New-ScheduledTaskAction -Execute $BatPath
$FullSettings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopOnIdleEnd `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 5) `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries

# Remove existing task if it exists
Unregister-ScheduledTask -TaskName $FullName -TaskPath $TaskFolder -Confirm:$false -ErrorAction SilentlyContinue

Register-ScheduledTask `
    -TaskName $FullName `
    -TaskPath $TaskFolder `
    -Trigger $FullTrigger `
    -Action $FullAction `
    -Settings $FullSettings `
    -Description "ETL + conjugate updates + game props + git push. Runs missed tasks on boot."

Write-Host "Created: $FullName (daily at 6 AM, runs on boot if missed)" -ForegroundColor Green

# ── Task 2: Game day props loop (every 30 min, 10 AM – 1 AM) ─
$LoopName = "TDD Game Day Props"
$LoopAction = New-ScheduledTaskAction -Execute $BatPath -Argument "--schedule-only"
$LoopSettings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopOnIdleEnd `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 5) `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30) `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew

# Trigger: every 30 min from 10 AM to 1 AM (15 hours)
$LoopTrigger = New-ScheduledTaskTrigger -Daily -At "10:00AM"
$LoopTrigger.Repetition = (New-ScheduledTaskTrigger -Once -At "10:00AM" `
    -RepetitionInterval (New-TimeSpan -Minutes 30) `
    -RepetitionDuration (New-TimeSpan -Hours 15)).Repetition

Unregister-ScheduledTask -TaskName $LoopName -TaskPath $TaskFolder -Confirm:$false -ErrorAction SilentlyContinue

Register-ScheduledTask `
    -TaskName $LoopName `
    -TaskPath $TaskFolder `
    -Trigger $LoopTrigger `
    -Action $LoopAction `
    -Settings $LoopSettings `
    -Description "Refresh schedule + lineups + game props every 30 min. Pushes to GitHub."

Write-Host "Created: $LoopName (every 30 min, 10 AM - 1 AM)" -ForegroundColor Green

Write-Host ""
Write-Host "Both tasks created in Task Scheduler under \TDD" -ForegroundColor Cyan
Write-Host "  - Full daily: 6 AM (ETL + projections + props + push)"
Write-Host "  - Game day:   every 30 min 10 AM-1 AM (schedule + props + push)"
Write-Host ""
Write-Host "Key settings:"
Write-Host "  - Runs missed tasks on boot (PC was off)"
Write-Host "  - Auto-retry on failure (3x for daily, 2x for loop)"
Write-Host "  - Runs on battery"
Write-Host "  - Loop skips if previous run still going"
