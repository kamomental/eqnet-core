"""Lightweight alerts evaluator/logger for ignition and ToM safety.

Emits JSON lines to `logs/alerts.jsonl` with a minimal schema:
{ "type": "ignite"|"autonomy.downshift"|"inhibit", "stage": str, "step": int, ... }
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable


@dataclass
class AlertsConfig:
    output: Path = Path("logs/alerts.jsonl")
    ignite_duration_warn_ms: int = 800
    ignite_duration_inhibit_ms: int = 1200
    tom_trust_threshold: float = 0.3


@dataclass
class AlertsLogger:
    config: AlertsConfig = field(default_factory=AlertsConfig)
    _buffer: List[Dict[str, Any]] = field(default_factory=list)
    downshift_fn: Optional[Callable[[], None]] = None

    def _write(self, payload: Dict[str, Any]) -> None:
        self.config.output.parent.mkdir(parents=True, exist_ok=True)
        with self.config.output.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _now(self) -> float:
        return time.time()

    def evaluate_and_log(self, episode: Dict[str, Any]) -> None:
        stage = episode.get("stage", "")
        step = int(episode.get("step", -1))
        ignite = episode.get("ignite", {}) if isinstance(episode, dict) else {}
        tom = episode.get("tom", {}) if isinstance(episode, dict) else {}

        # 1) Ignition trigger
        if ignite.get("trigger"):
            payload = {
                "type": "ignite",
                "ts": self._now(),
                "stage": stage,
                "step": step,
                "I": ignite.get("I"),
                "delta_R": ignite.get("delta_R"),
                "entropy_z": ignite.get("entropy_z"),
                "ignite_ms": ignite.get("ignite_ms"),
                "alerts": ignite.get("alerts"),
            }
            self._write(payload)

        # 2) Long ignition durations -> inhibit suggestion
        ignite_ms = ignite.get("ignite_ms")
        if isinstance(ignite_ms, (int, float)) and ignite_ms > self.config.ignite_duration_warn_ms:
            payload = {
                "type": "inhibit",
                "ts": self._now(),
                "stage": stage,
                "step": step,
                "ignite_ms": ignite_ms,
                "action": "cooldown" if ignite_ms < self.config.ignite_duration_inhibit_ms else "inhibit",
            }
            self._write(payload)
            # Also trigger a downshift on heavy duration
            if ignite_ms >= self.config.ignite_duration_inhibit_ms:
                ds = {
                    "type": "autonomy.downshift",
                    "ts": self._now(),
                    "stage": stage,
                    "step": step,
                    "intent_trust": tom.get("intent_trust", None),
                    "reason": "ignite_ms_heavy",
                }
                self._write(ds)
                if self.downshift_fn is not None:
                    try:
                        self.downshift_fn()
                    except Exception:
                        pass

        # 3) ToM low trust -> autonomy downshift
        trust = tom.get("intent_trust")
        if isinstance(trust, (int, float)) and trust < self.config.tom_trust_threshold:
            payload = {
                "type": "autonomy.downshift",
                "ts": self._now(),
                "stage": stage,
                "step": step,
                "intent_trust": trust,
            }
            self._write(payload)
            if self.downshift_fn is not None:
                try:
                    self.downshift_fn()
                except Exception:
                    pass


__all__ = ["AlertsLogger", "AlertsConfig"]
