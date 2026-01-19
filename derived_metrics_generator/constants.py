from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from common import sha256_hex, stable_json_dumps

_CFG_PATH = Path("config/derived_metrics.yaml")


def _load_cfg() -> Dict[str, Any]:
    if not _CFG_PATH.exists():
        raise FileNotFoundError(f"Missing config: {_CFG_PATH}")
    payload = yaml.safe_load(_CFG_PATH.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("derived_metrics config must be a mapping")
    return payload


_CFG = _load_cfg()

# Stable hash of the config file (for integrity tagging).
_CFG_HASH = sha256_hex(
    stable_json_dumps(_CFG)
)

# Fixed window presets (not configurable via CLI).
WINDOWS_MS = dict(_CFG.get("windows_ms") or _CFG.get("windows") or {})

# Metric coverage per window (fixed).
METRIC_WINDOWS = {
    key: set(value or [])
    for key, value in (_CFG.get("metric_windows") or _CFG.get("metrics") or {}).items()
}

DEFAULT_CALC_VERSION = str(
    _CFG.get("default_calc_version")
    or _CFG.get("calc_version")
    or "v1.0.0"
)

# Numeric stability for correlations.
EPS = float(_CFG.get("eps", (_CFG.get("numeric_stability") or {}).get("eps", 1e-6)))

# Public tag used in derived_metrics output to bind config lineage.
CFG_HASH_TAG = _CFG_HASH[:12]
