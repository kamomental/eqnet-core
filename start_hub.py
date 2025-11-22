#!/usr/bin/env python
"""Entry-point to launch the EQNet hub without installing the package."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ROOT_PKG = ROOT / "emot_terrain_lab"
if ROOT_PKG.exists():
    pkg_path = str(ROOT_PKG)
    if pkg_path not in sys.path:
        sys.path.insert(0, pkg_path)

if __name__ == "__main__":
    importlib.import_module("emot_terrain_lab.hub.runtime")