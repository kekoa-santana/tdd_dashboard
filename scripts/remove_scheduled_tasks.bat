@echo off
REM  Remove TDD Dashboard scheduled tasks. Run as Administrator.

echo Removing TDD scheduled tasks...
schtasks /delete /tn "TDD Morning Update" /f 2>nul
schtasks /delete /tn "TDD Hourly Update" /f 2>nul
echo Done.
pause
