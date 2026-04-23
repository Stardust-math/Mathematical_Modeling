@echo off
setlocal
cd /d "%~dp0"

call tools\ui_helpers.bat banner "Curve Fitting Project - Main Experiments"
call tools\ui_helpers.bat progress 10 "Checking conda..."

where conda >nul 2>nul
if errorlevel 1 (
    call tools\ui_helpers.bat error "Conda not found in PATH."
    pause
    exit /b 1
)

call tools\ui_helpers.bat progress 20 "Checking experiment script..."
if not exist scripts\run_main_experiments.py (
    call tools\ui_helpers.bat error "scripts\run_main_experiments.py not found."
    pause
    exit /b 1
)

call tools\ui_helpers.bat progress 35 "Preparing output directories..."
call tools\ui_helpers.bat progress 55 "Running reconstruction experiments..."
call tools\ui_helpers.bat progress 75 "Saving figures and csv files..."
call conda run -n curvefit_env python scripts/run_main_experiments.py
if errorlevel 1 (
    call tools\ui_helpers.bat error "Main experiments failed."
    pause
    exit /b 1
)

call tools\ui_helpers.bat progress 100 "Main experiments finished."
call tools\ui_helpers.bat ok "Results were written to outputs\results and outputs\figures."
pause
