#!/usr/bin/env bash
# Fast-path/Nightly audit helper
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python "$SCRIPT_DIR/emot_terrain_lab/scripts/run_audit.py" "$@"