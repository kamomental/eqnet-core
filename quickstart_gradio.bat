@echo off
REM Launch the Gradio demo with current profile
pushd %~dp0
python gradio_demo_prev.py %*
popd
