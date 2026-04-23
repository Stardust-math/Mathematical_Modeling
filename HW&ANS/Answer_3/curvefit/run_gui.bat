@echo off
setlocal
cd /d "%~dp0"

call tools\ui_helpers.bat banner "Curve Fitting Project - Streamlit GUI"
call tools\ui_helpers.bat progress 10 "Checking conda..."

where conda >nul 2>nul
if errorlevel 1 (
    call tools\ui_helpers.bat error "Conda not found in PATH."
    pause
    exit /b 1
)

call tools\ui_helpers.bat progress 25 "Checking GUI entry..."
if not exist app\streamlit_app.py (
    call tools\ui_helpers.bat error "app\streamlit_app.py not found."
    pause
    exit /b 1
)

call tools\ui_helpers.bat progress 50 "Preparing Streamlit service..."
call tools\ui_helpers.bat progress 75 "Launching browser session..."
call tools\ui_helpers.bat progress 100 "GUI is starting..."
call conda run -n curvefit_env streamlit run app/streamlit_app.py
if errorlevel 1 (
    call tools\ui_helpers.bat error "Failed to launch GUI."
    pause
    exit /b 1
)
pause
