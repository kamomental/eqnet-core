@echo off
REM Launch the full Gradio demo (not the primary core quickstart)
pushd %~dp0
IF EXIST .venv\Scripts\activate.bat (CALL .venv\Scripts\activate.bat)
echo [info] quickstart_gradio.bat is the full demo path. Use quickstart_core.bat for the primary EQNet core loop.
python gradio_demo_prev.py %*
popd

