@echo off
REM Launch the Streamlit telemetry viewer (auto-activates .venv if present)
pushd %~dp0
IF EXIST .venv\Scripts\activate.bat (CALL .venv\Scripts\activate.bat)
streamlit run tools\eqnet_telemetry_viewer.py %*
popd

