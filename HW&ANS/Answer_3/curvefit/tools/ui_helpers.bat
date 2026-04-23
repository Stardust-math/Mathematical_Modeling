@echo off
setlocal EnableDelayedExpansion

if /I "%~1"=="banner" goto banner
if /I "%~1"=="progress" goto progress
if /I "%~1"=="ok" goto ok
if /I "%~1"=="error" goto error
exit /b 0

:banner
echo.
echo ============================================================
echo   %~2
echo ============================================================
echo.
exit /b 0

:progress
set "PERCENT=%~2"
set "MESSAGE=%~3"
if "%PERCENT%"=="" set "PERCENT=0"
if "%MESSAGE%"=="" set "MESSAGE=Processing..."

set /a FILLED=%PERCENT%/5
set "BAR="
for /L %%i in (1,1,20) do (
    if %%i LEQ !FILLED! (
        set "BAR=!BAR!#"
    ) else (
        set "BAR=!BAR!-"
    )
)
echo [!BAR!] %PERCENT%%%  %MESSAGE%
exit /b 0

:ok
echo [DONE] %~2
exit /b 0

:error
echo [ERROR] %~2
exit /b 1
