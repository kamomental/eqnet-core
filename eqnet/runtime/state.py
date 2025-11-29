"""Lightweight containers for runtime qualia state."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class QualiaState:
    """Snapshot of the qualia field at a single moment.

    Notes
    -----
    - ``qualia_vec`` is always coerced to a 1-D numpy array (shape: (D,)).
      Batched / multi-sample processing should live in a separate analysis
      layer when we eventually support it.
    - ``membrane_state`` / ``flux`` / ``narrative_ref`` are reserved for
      terrain / fog / danger metadata, flows, and story linkage.
    """

    timestamp: datetime
    qualia_vec: np.ndarray
    membrane_state: Dict[str, Any] = field(default_factory=dict)
    flux: Dict[str, Any] = field(default_factory=dict)
    narrative_ref: Optional[str] = None

    def __post_init__(self) -> None:
        self.qualia_vec = np.asarray(self.qualia_vec, dtype=float).reshape(-1)
