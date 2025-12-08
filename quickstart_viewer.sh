#!/usr/bin/env bash
# Launch the Streamlit telemetry viewer
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
streamlit run tools/eqnet_telemetry_viewer.py "$@"
