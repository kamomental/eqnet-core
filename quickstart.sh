#!/usr/bin/env bash
# Quick start helper for macOS/Linux users
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python "$SCRIPT_DIR/emot_terrain_lab/scripts/run_quickstart.py" "$@"
