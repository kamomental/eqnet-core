"""Repair controller handling rollbacks when critical thresholds break."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from emot_terrain_lab.eqcore.state import CoreState
from runtime.checkpoint import CheckpointStore


@dataclass
class RepairMetrics:
    """Metrics observed by the repair controller."""

    rho: float
    synchrony: float
    misfires: int = 0
    exceptions: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class RepairConfig:
    """Thresholds controlling automatic repair."""

    rho_max: float = 1.8
    synchrony_max: float = 0.78
    max_misfires: int = 3
    cooldown_seconds: float = 480.0
    rewind_steps: int = 3


class RepairController:
    """Detect anomalies and rewind state using checkpoints."""

    def __init__(self, store: CheckpointStore[CoreState], config: Optional[RepairConfig] = None) -> None:
        self.store = store
        self.config = config or RepairConfig()
        self._last_repair_ts: float = 0.0

    def should_trigger(self, metrics: RepairMetrics) -> Optional[str]:
        if time.time() - self._last_repair_ts < self.config.cooldown_seconds:
            return None
        if metrics.rho > self.config.rho_max:
            return f"rho>{self.config.rho_max:.2f}"
        if metrics.synchrony > self.config.synchrony_max:
            return f"R>{self.config.synchrony_max:.2f}"
        if metrics.misfires >= self.config.max_misfires:
            return "misfires"
        if metrics.exceptions > 0:
            return "exception"
        return None

    def repair(self, metrics: RepairMetrics) -> Optional[CoreState]:
        reason = self.should_trigger(metrics)
        if reason is None:
            return None
        record = self.store.rewind(self.config.rewind_steps)
        if record is None:
            return None
        self._last_repair_ts = metrics.timestamp
        return record.state
