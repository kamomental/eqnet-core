@echo off
setlocal
cd /d %~dp0\..
set PYTHONPATH=.
python tools\opc_runner.py --reset %*
endlocal
