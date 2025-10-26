@echo off
REM Launch EQNet local bus, watcher, hotkeys, and dashboard in parallel.
setlocal
pushd %~dp0
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "emot_terrain_lab\scripts\start_local_stack.ps1" %*
popd
endlocal
