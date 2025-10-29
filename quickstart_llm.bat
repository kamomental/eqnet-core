@echo off
setlocal enabledelayedexpansion

REM Load environment variables from .env if present
if exist ".env" (
  for /f "usebackq tokens=1* delims==" %%A in (".env") do (
    set "key=%%A"
    set "val=%%B"
    if defined key (
      set "key=!key: =!"
      if not "!key!"=="" (
        set "!key!=!val!"
      )
    )
  )
)

echo [0/5] Checking available LLM models (custom endpoint or LM Studio)...
python scripts\list_llm_models.py
if errorlevel 1 (
  echo   (hint: start LM Studio's OpenAI-compatible server or ignore if you only need the terrain quickstart)
)

echo [1/5] Installing dependencies (requirements-dev.txt)...
pip install -r requirements-dev.txt
if errorlevel 1 goto :error

echo [2/5] Running quick loop replay (field metrics -> ignition)...
python scripts\run_quick_loop.py --field_metrics_log data\field_metrics.jsonl --steps 200 %*
if errorlevel 1 goto :error

echo [3/5] Generating nightly report (care / canary / value influence)...
python -m emot_terrain_lab.ops.nightly --telemetry_log "telemetry\ignition-*.jsonl"
if errorlevel 1 goto :error

echo [4/5] Generating monthly value influence highlights...
python scripts\gen_monthly_highlights.py
if errorlevel 1 goto :error

echo [5/5] Showing EQNet vs plain LLM comparison...
python scripts\demo_eqnet_vs_llm.py
if errorlevel 1 (
  echo   (comparison demo failed; ensure LM Studio server is reachable and try `python scripts\demo_eqnet_vs_llm.py` manually)
)

echo.
echo Quick start completed. Artifacts live under telemetry\*, reports\nightly.*, reports\monthly\*.
echo See above for EQNet vs LLM response summary. For visuals run: python emot_terrain_lab/scripts/gradio_demo.py
goto :eof

:error
echo.
echo Quick start failed. Check the messages above for details.
exit /b 1
