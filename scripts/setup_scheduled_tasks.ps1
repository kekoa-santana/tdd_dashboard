# TDD Dashboard: Windows Task Scheduler setup.
# Run once from an Administrator PowerShell:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_scheduled_tasks.ps1
#
# Leaves exactly two active pipeline tasks:
#   TDD Full Daily Update   6:00 AM daily
#       ETL, precompute, projections, full game sims, publish.
#   TDD Intraday Refresh    every 15 minutes, 9:00 AM to midnight
#       Re-sims only games whose starters, umpire, or lineups changed,
#       refreshes live standouts, publishes.
#
# Superseded tasks are disabled, not deleted, so any of them can be
# re-enabled from Task Scheduler.

$ErrorActionPreference = 'Stop'
$failures = 0

function Invoke-Step([string]$label, [scriptblock]$body) {
    # Each step stands alone so one failure cannot block the rest.
    try { & $body; Write-Host "OK    $label" }
    catch { $script:failures++; Write-Host "FAIL  $label : $($_.Exception.Message)" }
}

$project = 'C:\Users\kekoa\Documents\data_analytics\tdd-dashboard'
$bat = Join-Path $project 'scripts\daily_update.bat'
$superseded = @(
    'TDD Morning Update',
    'TDD Hourly Update',
    'TDD Game Day Props',
    'MLB_DailyRefresh_Morning',
    'MLB_DailyRefresh_Afternoon'
)

function Get-Task([string]$name) {
    Get-ScheduledTask | Where-Object { $_.TaskName -eq $name } | Select-Object -First 1
}

# Tasks run as the current user, only when signed in, like the originals.
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

# Never let the scheduler hard-kill a run mid-write: a killed run leaves the
# pipeline lock behind and every later run skips until it goes stale.
function New-PipelineSettings([TimeSpan]$limit) {
    New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit $limit `
        -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
}

# -- Morning run ---------------------------------------------------------
$morning = Get-Task 'TDD Full Daily Update'
$morningAction = New-ScheduledTaskAction -Execute $bat -WorkingDirectory $project
$morningTrigger = New-ScheduledTaskTrigger -Daily -At 6:00AM
if ($morning) {
    # Edit the existing definition in place. Replacing its whole settings
    # block drops fields like the RestartOnFailure count, which the scheduler
    # rejects ("task XML is missing a required element").
    Invoke-Step 'TDD Full Daily Update: 3 hour time limit' {
        $morning.Settings.ExecutionTimeLimit = 'PT3H'
        Set-ScheduledTask -InputObject $morning | Out-Null
    }
} else {
    Register-ScheduledTask -TaskName 'TDD Full Daily Update' -Action $morningAction -Trigger $morningTrigger `
        -Settings (New-PipelineSettings (New-TimeSpan -Hours 3)) -Principal $principal `
        -Description 'ETL, precompute, projections, full game sims, publish to R2.' | Out-Null
}

# -- Intraday refresh ----------------------------------------------------
$intradayAction = New-ScheduledTaskAction -Execute $bat -Argument '--intraday' -WorkingDirectory $project
$intradayTrigger = New-ScheduledTaskTrigger -Daily -At 9:00AM
$intradayTrigger.Repetition = (New-ScheduledTaskTrigger -Once -At 9:00AM `
    -RepetitionInterval (New-TimeSpan -Minutes 15) -RepetitionDuration (New-TimeSpan -Hours 15)).Repetition
$intraday = Get-Task 'TDD Intraday Refresh'
if ($intraday) {
    Write-Host "OK    TDD Intraday Refresh already exists"
} else {
    Register-ScheduledTask -TaskName 'TDD Intraday Refresh' -Action $intradayAction -Trigger $intradayTrigger `
        -Settings (New-PipelineSettings (New-TimeSpan -Hours 1)) -Principal $principal `
        -Description 'Re-sims games whose starters, umpire, or lineups changed; live standouts; publish to R2.' | Out-Null
}

# -- Disable superseded tasks ----------------------------------------------
foreach ($name in $superseded) {
    $task = Get-Task $name
    if ($task -and $task.State -ne 'Disabled') {
        Invoke-Step "Disable $name" {
            Disable-ScheduledTask -TaskName $task.TaskName -TaskPath $task.TaskPath | Out-Null
        }
    }
}

Get-ScheduledTask | Where-Object { $_.TaskName -match '^(TDD|MLB_)' } |
    Sort-Object State, TaskName |
    Format-Table TaskName, State, @{ n = 'Limit'; e = { $_.Settings.ExecutionTimeLimit } } -AutoSize

if ($failures -gt 0) { Write-Host "$failures step(s) failed"; exit 1 }
