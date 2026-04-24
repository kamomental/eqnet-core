#!/usr/bin/env bash
# Launch the full Gradio demo (not the primary core quickstart)
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi
echo "[info] quickstart_gradio.sh is the full demo path. Use quickstart_core.sh for the primary EQNet core loop."
python gradio_demo_prev.py "$@"
