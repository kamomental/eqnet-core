@echo off
REM Launch the Gradio demo with current profile (auto-activates .venv if present)
pushd %~dp0
IF EXIST .venv\Scripts\activate.bat (CALL .venv\Scripts\activate.bat)
python gradio_demo_prev.py %*
popd

