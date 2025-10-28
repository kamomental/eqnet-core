"""Light-weight validation helpers."""

from __future__ import annotations

from typing import Any, Dict, Tuple


def validate_pain_record(rec: Dict[str, Any]) -> Tuple[bool, str | None]:
    required = ["ts_ms", "kind", "delta_aff", "reasons", "hp", "context"]
    for key in required:
        if key not in rec:
            return False, f"missing:{key}"
    if not isinstance(rec["ts_ms"], int):
        return False, "ts_ms"
    if not isinstance(rec["delta_aff"], (int, float)):
        return False, "delta_aff"
    if rec["delta_aff"] >= 0:
        return False, "delta_aff_not_negative"
    if not isinstance(rec["reasons"], list):
        return False, "reasons"
    hp = rec.get("hp", {})
    if not isinstance(hp, dict):
        return False, "hp"
    for part in ("emotional", "metabolic", "ethical"):
        if part not in hp:
            return False, f"hp_missing:{part}"
    return True, None


__all__ = ["validate_pain_record"]
