@echo off
setlocal
cd /d "%~dp0"

call tools\ui_helpers.bat banner "Curve Fitting Project - Figure Generation"
call tools\ui_helpers.bat progress 10 "Checking conda..."

where conda >nul 2>nul
if errorlevel 1 (
    call tools\ui_helpers.bat error "Conda not found in PATH."
    pause
    exit /b 1
)

call tools\ui_helpers.bat progress 20 "Checking figure script..."
if not exist scripts\generate_paper_figures.py (
    call tools\ui_helpers.bat error "scripts\generate_paper_figures.py not found."
    pause
    exit /b 1
)

call tools\ui_helpers.bat progress 40 "Loading existing experiment csv files..."
call tools\ui_helpers.bat progress 65 "Drawing publication-style charts..."
call tools\ui_helpers.bat progress 85 "Saving figure files..."
call conda run -n curvefit_env python scripts/generate_paper_figures.py
if errorlevel 1 (
    call tools\ui_helpers.bat error "Figure generation failed."
    pause
    exit /b 1
)

call tools\ui_helpers.bat progress 100 "Figure generation finished."
call tools\ui_helpers.bat ok "Figures were exported to outputs\figures."
pause
