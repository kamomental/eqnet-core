@echo off
setlocal

pushd %~dp0\..\..

for /f "delims=" %%i in ('powershell -NoProfile -Command "Get-ChildItem ..\\..\\trace_runs -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty Name"') do set RUN_ID=%%i
if "%RUN_ID%"=="" (
  echo [error] trace_runs has no run_id folders.
  popd
  exit /b 1
)

for /f "delims=" %%j in ('powershell -NoProfile -Command "Get-ChildItem ..\\..\\trace_runs\\%RUN_ID% -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty Name"') do set DAY_DIR=%%j
if "%DAY_DIR%"=="" (
  echo [error] no day folder found under trace_runs\%RUN_ID%.
  popd
  exit /b 1
)

python docs\replay\make_replay.py --trace_dir trace_runs\%RUN_ID%\%DAY_DIR% --out docs\replay\replay.json
if errorlevel 1 (
  echo [error] failed to build replay.json
  popd
  exit /b 1
)

echo [OK] replay.json built from %RUN_ID%\%DAY_DIR%
echo [next] http://localhost:8000/docs/replay/replay.html
python -m http.server 8000

popd
