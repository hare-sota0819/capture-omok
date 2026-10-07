@echo off
rem Double-click once: installs numba (the library that compiles the agent's search).
title install numba
cd /d "%~dp0"
set PY=
where py >nul 2>nul
if not errorlevel 1 set PY=py
if "%PY%"=="" (
    where python >nul 2>nul
    if not errorlevel 1 set PY=python
)
if "%PY%"=="" (
    echo Python was not found on this computer.
    pause
    exit /b 1
)
echo Installing numba with %PY% ...
%PY% -m pip install --upgrade -r requirements.txt
echo.
%PY% -c "import sys, numba, numpy; print('python', sys.version.split()[0], 'numba', numba.__version__, 'numpy', numpy.__version__)"
echo.
echo Done. You can close this window.
pause
