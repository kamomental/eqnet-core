@echo off
REM Fast-path/Nightly audit helper
pushd %~dp0
python emot_terrain_lab\scripts\run_audit.py %*
pause

