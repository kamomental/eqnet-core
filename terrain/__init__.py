"""Compatibility shim so ``import terrain`` resolves to ``emot_terrain_lab.terrain``."""
from __future__ import annotations

import importlib
import sys

_real_pkg = importlib.import_module("emot_terrain_lab.terrain")
# Copy its attributes into this module's globals for convenience.
globals().update(_real_pkg.__dict__)
# Make sure `terrain` (this package) and its submodules point to the real package.
sys.modules[__name__] = _real_pkg