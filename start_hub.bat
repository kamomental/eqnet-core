@echo off
REM Simple wrapper to launch the EQNet hub with the correct PYTHONPATH.
set SCRIPT_DIR=%~dp0
set PYTHONPATH=%SCRIPT_DIR%
python "%SCRIPT_DIR%start_hub.py" %*