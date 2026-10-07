@echo off
rem Double-click to open the game window.
cd /d "%~dp0game"
where py >nul 2>nul
if not errorlevel 1 (
    py omok.py %*
    goto finished
)
where python >nul 2>nul
if not errorlevel 1 (
    python omok.py %*
    goto finished
)
echo Python was not found on this computer.
echo Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH".
pause
exit /b 1
:finished
if errorlevel 1 pause
