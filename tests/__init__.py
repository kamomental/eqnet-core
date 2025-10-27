"""Test package safeguards."""

from __future__ import annotations

import sys
import types

try:  # pragma: no cover - normal path when pytest is installed
    import pytest as _pytest  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for limited environments
    shim = types.ModuleType("pytest")

    def _missing(*args, **kwargs):
        raise RuntimeError("pytest is required to run the test suite. Install it via `pip install pytest`.")

    shim.approx = _missing
    shim.mark = types.SimpleNamespace(skip=_missing)
    sys.modules["pytest"] = shim
else:
    sys.modules["pytest"] = _pytest
