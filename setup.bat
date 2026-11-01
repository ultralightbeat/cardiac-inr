@echo off
rem Double-click or run from cmd. Add -Mirror for the Tsinghua pip mirror, -Quick for a 5-minute check.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
pause
