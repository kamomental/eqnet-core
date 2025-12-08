@echo off
REM Launch the Streamlit telemetry viewer
pushd %~dp0
IF NOT DEFINED STREAMLIT_BROWSER GOTO run
SET STREAMLIT_BROWSER=%STREAMLIT_BROWSER%
:run
streamlit run tools\eqnet_telemetry_viewer.py %*
popd
