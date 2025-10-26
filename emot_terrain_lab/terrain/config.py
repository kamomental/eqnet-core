# -*- coding: utf-8 -*-
"""Configuration helpers for cultural projection matrices and engine settings."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import yaml

from .emotion import AXES


class CultureProjection:
    """Load and serve culture-specific projection matrices."""

    def __init__(self, table: Dict[str, Dict[str, Iterable[float]]]) -> None:
        self._table = table
        self._validate()

    def _validate(self) -> None:
        expected = len(AXES)
        for locale, axes in self._table.items():
            if "axis1" not in axes or "axis2" not in axes:
                raise ValueError(f"Culture '{locale}' must define axis1 and axis2 vectors.")
            for axis_name in ("axis1", "axis2"):
                vec = list(axes[axis_name])
                if len(vec) != expected:
                    raise ValueError(
                        f"Culture '{locale}' axis '{axis_name}' must have length {expected}, got {len(vec)}."
                    )

    def matrix(self, locale: str) -> np.ndarray:
        entry = self._table.get(locale) or self._table.get("default")
        if entry is None:
            raise KeyError(f"Projection for locale '{locale}' or 'default' not found.")
        mat = np.vstack([entry["axis1"], entry["axis2"]])
        return mat.astype(float)


@functools.lru_cache(maxsize=4)
def load_culture_projection(path: Path | str = Path("config/culture.yaml")) -> CultureProjection:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    table = data.get("culture_projection")
    if not isinstance(table, dict):
        raise ValueError("culture_projection key is missing or invalid in configuration.")
    return CultureProjection(table)


def project_emotion(z: np.ndarray, locale: str, path: Path | str = Path("config/culture.yaml")) -> np.ndarray:
    """Project 9D emotion vector onto 2D cultural plane."""
    proj = load_culture_projection(path)
    mat = proj.matrix(locale)
    return mat @ z
