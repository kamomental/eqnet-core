"""Minimal MAP-Elites archive manager."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


@dataclass
class Elite:
    fitness: float
    descriptor: Tuple[float, float]
    payload: Dict[str, float]


@dataclass
class MapElitesArchive:
    grid_shape: Tuple[int, int]
    descriptor_ranges: Tuple[Tuple[float, float], Tuple[float, float]]
    archive: Dict[Tuple[int, int], Elite] = field(default_factory=dict)

    def update(self, fitness: float, descriptor: Tuple[float, float], payload: Dict[str, float]) -> bool:
        cell = self._cell_index(descriptor)
        if cell not in self.archive or fitness > self.archive[cell].fitness:
            self.archive[cell] = Elite(fitness, descriptor, payload)
            return True
        return False

    def _cell_index(self, descriptor: Tuple[float, float]) -> Tuple[int, int]:
        idx = []
        for i, (value, (lo, hi)) in enumerate(zip(descriptor, self.descriptor_ranges)):
            value = np.clip(value, lo, hi)
            frac = (value - lo) / (hi - lo + 1e-8)
            idx.append(min(self.grid_shape[i] - 1, int(frac * self.grid_shape[i])))
        return tuple(idx)

    def export(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "cell": list(cell),
                "fitness": elite.fitness,
                "descriptor": list(elite.descriptor),
                "payload": elite.payload,
            }
            for cell, elite in self.archive.items()
        ]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
