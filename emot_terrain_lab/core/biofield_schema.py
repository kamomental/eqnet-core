# -*- coding: utf-8 -*-
"""Lightweight validation for biofield configuration files."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

REQUIRED_KEYS = {"version", "nodes", "adjacency", "decay", "coupling", "limits"}


def validate_bio_cfg(cfg: Mapping[str, Any]) -> List[str]:
    """Return a list of human-readable validation warnings."""
    errors: List[str] = []
    missing = REQUIRED_KEYS - set(cfg.keys())
    if missing:
        errors.append(f"missing keys: {sorted(missing)}")

    nodes = cfg.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        errors.append("nodes must be a non-empty list")

    decay = cfg.get("decay", {})
    for key in ("alpha", "beta_inject"):
        val = decay.get(key)
        if not isinstance(val, (int, float)):
            errors.append(f"decay.{key} must be numeric")

    adjacency = cfg.get("adjacency", {})
    if isinstance(adjacency, Mapping):
        for edge, weight in adjacency.items():
            if isinstance(edge, str):
                parts = edge.split("->")
                if len(parts) != 2 or not parts[0] or not parts[1]:
                    errors.append(f"invalid adjacency key '{edge}'")
            elif not (
                isinstance(edge, (list, tuple))
                and len(edge) == 2
                and all(isinstance(p, str) for p in edge)
            ):
                errors.append(f"invalid adjacency entry '{edge}'")
            if not isinstance(weight, (int, float)):
                errors.append(f"adjacency weight for '{edge}' must be numeric")

    coupling = cfg.get("coupling", {})
    if isinstance(coupling, Mapping):
        for src, mapping in coupling.items():
            if not isinstance(mapping, Mapping):
                errors.append(f"coupling.{src} must be mapping")
                continue
            for dst, spec in mapping.items():
                if isinstance(spec, Mapping):
                    gain = spec.get("gain")
                    if not isinstance(gain, (int, float)):
                        errors.append(f"coupling.{src}.{dst}.gain must be numeric")
                elif not isinstance(spec, (int, float)):
                    errors.append(f"coupling.{src}.{dst} must be numeric or mapping")

    limits = cfg.get("limits", {})
    if isinstance(limits, Mapping):
        for key, bounds in limits.items():
            if key == "jerk":
                continue
            if not (
                isinstance(bounds, (list, tuple))
                and len(bounds) == 2
                and all(isinstance(x, (int, float)) for x in bounds)
            ):
                errors.append(f"limits.{key} must be [lo, hi]")
            elif bounds[0] >= bounds[1]:
                errors.append(f"limits.{key} must satisfy lo < hi")
    return errors


__all__ = ["validate_bio_cfg"]
