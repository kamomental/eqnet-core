#!/usr/bin/env bash
# Primary quickstart for the EQNet core loop
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export UV_CACHE_DIR="$SCRIPT_DIR/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$SCRIPT_DIR/.uv-python"
uv run python "$SCRIPT_DIR/scripts/core_quickstart_demo.py" "$@"
