@echo off
setlocal enabledelayedexpansion

:: --- Configuration ---
set "PYTHON_SCRIPT=ai34.py"
set "PROMPTS_FILE=aiprompts.txt" # <<< Name of the file AIs might modify
set "PYTHON_EXE=python"
set "SCRIPT_DIR=%~dp0"
if not "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR%\"
:: --- End Configuration ---

:: --- Basic Checks ---
if not exist "%SCRIPT_DIR%%PYTHON_SCRIPT%" ( /* ... */ pause; exit /b 1)
if not exist "%SCRIPT_DIR%%PROMPTS_FILE%" ( /* ... */ pause; exit /b 1)
:: --- End Checks ---

:: --- Argument Forwarding ---
set "FORWARDED_ARGS=%*" # <<< Use the simpler forwarding that worked
:: --- End Argument Forwarding ---

echo Starting prompt processing...
echo Python Script: "%SCRIPT_DIR%%PYTHON_SCRIPT%"
echo Prompts File: "%SCRIPT_DIR%%PROMPTS_FILE%" # <<< This file can now be modified by AI
if defined FORWARDED_ARGS ( echo Arguments being forwarded: !FORWARDED_ARGS! ) else ( echo No args to forward. )
echo ==========================================================

for /f "usebackq delims=" %%P in ("%SCRIPT_DIR%%PROMPTS_FILE%") do (
    set "CURRENT_PROMPT=%%P"
    if not defined CURRENT_PROMPT ( echo Skipping empty line. ) else (
        echo.
        echo --- Processing Prompt: "!CURRENT_PROMPT!" ---
        echo.

        :: Construct the command - ADD --prompts_file argument
        set "PYTHON_CMD=%PYTHON_EXE% "%SCRIPT_DIR%%PYTHON_SCRIPT%" --prompt "!CURRENT_PROMPT!" --prompts_file "%SCRIPT_DIR%%PROMPTS_FILE%" !FORWARDED_ARGS!"
        # ^^^ ADDED: --prompts_file "%SCRIPT_DIR%%PROMPTS_FILE%" ^^^

        echo DEBUG: Running -> !PYTHON_CMD!
        call !PYTHON_CMD!

        if errorlevel 1 ( echo WARNING: Python errorlevel !errorlevel!; pause )
    )
)

echo ==========================================================
echo All prompts processed.
endlocal
echo.
pause
