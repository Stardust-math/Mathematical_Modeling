@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "ENV_NAME=hw4-sir-periodic"
set "MODE=full"
if not "%~1"=="" set "MODE=%~1"

echo ============================================
echo HW4 SIR Periodic Outbreak - Run All
echo Mode: %MODE%
echo ============================================

where conda >nul 2>nul
if %ERRORLEVEL%==0 (
    if not exist "environment.yml" (
        echo ERROR: environment.yml was not found.
        pause
        exit /b 1
    )
    set "FOUND_ENV="
    for /f "delims=" %%E in ('conda env list ^| findstr /R /C:"^%ENV_NAME%[ ]"') do set "FOUND_ENV=1"
    if defined FOUND_ENV (
        echo Conda environment "%ENV_NAME%" already exists. Skipping creation.
    ) else (
        echo Creating conda environment "%ENV_NAME%" from environment.yml ...
        conda env create -f "environment.yml"
        if errorlevel 1 goto fail
    )
    conda run -n "%ENV_NAME%" python "code\main_all.py" --mode "%MODE%"
    if errorlevel 1 goto fail
    goto success
)

python "code\main_all.py" --mode "%MODE%"
if errorlevel 1 goto fail

goto success

:success
echo.
echo Done. Figures are in figs\ and CSV files are in results\.
pause
exit /b 0

:fail
echo.
echo ERROR: Experiment run failed.
pause
exit /b 1
