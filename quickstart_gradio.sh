#!/usr/bin/env bash
# Launch the Gradio demo helper
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
python gradio_demo_prev.py "$@"
