@echo off
setlocal
cd /d "%~dp0"

call tools\ui_helpers.bat banner "Curve Fitting Project - Smoke Test"
call tools\ui_helpers.bat progress 10 "Checking conda..."

where conda >nul 2>nul
if errorlevel 1 (
    call tools\ui_helpers.bat error "Conda not found in PATH."
    pause
    exit /b 1
)

call tools\ui_helpers.bat progress 25 "Checking project files..."
if not exist scripts\smoke_test.py (
    call tools\ui_helpers.bat error "scripts\smoke_test.py not found."
    pause
    exit /b 1
)

call tools\ui_helpers.bat progress 45 "Starting smoke test..."
call tools\ui_helpers.bat progress 70 "Running core module checks..."
call conda run -n curvefit_env python scripts/smoke_test.py
if errorlevel 1 (
    call tools\ui_helpers.bat error "Smoke test failed."
    pause
    exit /b 1
)

call tools\ui_helpers.bat progress 100 "Smoke test finished."
call tools\ui_helpers.bat ok "Project core modules passed the quick check."
pause
