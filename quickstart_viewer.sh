#!/usr/bin/env bash
# Launch the Streamlit telemetry viewer (auto-activates .venv if present)
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi
streamlit run tools/eqnet_telemetry_viewer.py "$@"
