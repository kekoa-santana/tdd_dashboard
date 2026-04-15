@echo off
schtasks /query /tn "TDD Hourly Update" /v /fo LIST
echo ===MORNING===
schtasks /query /tn "TDD Morning Update" /v /fo LIST
