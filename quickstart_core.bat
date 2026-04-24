@echo off
REM Primary quickstart for the EQNet core loop
pushd %~dp0
set "UV_CACHE_DIR=%CD%\.uv-cache"
set "UV_PYTHON_INSTALL_DIR=%CD%\.uv-python"
uv run python scripts\core_quickstart_demo.py %*
popd
