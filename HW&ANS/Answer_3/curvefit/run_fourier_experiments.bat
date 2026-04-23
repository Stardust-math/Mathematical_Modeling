@echo off
setlocal
cd /d "%~dp0"

call tools\ui_helpers.bat banner "Curve Fitting Project - Fourier Experiments"
call tools\ui_helpers.bat progress 10 "Checking conda..."

where conda >nul 2>nul
if errorlevel 1 (
    call tools\ui_helpers.bat error "Conda not found in PATH."
    pause
    exit /b 1
)

call tools\ui_helpers.bat progress 20 "Checking Fourier experiment script..."
if not exist scripts\run_fourier_experiments.py (
    call tools\ui_helpers.bat error "scripts\run_fourier_experiments.py not found."
    pause
    exit /b 1
)

call tools\ui_helpers.bat progress 35 "Preparing Fourier output directories..."
call tools\ui_helpers.bat progress 55 "Running K comparison and spectrum analysis..."
call tools\ui_helpers.bat progress 75 "Exporting epicycle keyframes and animations..."
call conda run -n curvefit_env python scripts/run_fourier_experiments.py
if errorlevel 1 (
    call tools\ui_helpers.bat error "Fourier experiments failed."
    pause
    exit /b 1
)

call tools\ui_helpers.bat progress 100 "Fourier experiments finished."
call tools\ui_helpers.bat ok "Fourier results were written to outputs folders."
pause
