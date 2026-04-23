@echo off
setlocal
cd /d "%~dp0"

call tools\ui_helpers.bat banner "Curve Fitting Project - Environment Setup"
call tools\ui_helpers.bat progress 5 "Checking conda..."

where conda >nul 2>nul
if errorlevel 1 (
    call tools\ui_helpers.bat error "Conda not found in PATH. Please open Anaconda Prompt first."
    pause
    exit /b 1
)

call tools\ui_helpers.bat progress 15 "Reading environment.yml..."
if not exist environment.yml (
    call tools\ui_helpers.bat error "environment.yml not found."
    pause
    exit /b 1
)

call tools\ui_helpers.bat progress 35 "Preparing environment curvefit_env..."
call tools\ui_helpers.bat progress 55 "Solving package dependencies..."
call tools\ui_helpers.bat progress 75 "Installing or updating packages..."
call conda env update -f environment.yml --prune
if errorlevel 1 (
    call tools\ui_helpers.bat error "Failed to create or update environment."
    pause
    exit /b 1
)

call tools\ui_helpers.bat progress 100 "Environment ready."
call tools\ui_helpers.bat ok "curvefit_env is available now."
pause
