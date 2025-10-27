from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class IgnitionCfg:
    theta_on: float = field(default=0.62)
    theta_off: float = field(default=0.48)
    dwell_steps: int = field(default=8)


@dataclass
class TelemetryCfg:
    log_path: str = field(default="telemetry/ignition-%Y%m%d.jsonl")


@dataclass
class ReplayCfg:
    min_interval_ms: int = field(default=10)
    sample_every: int = field(default=1)


@dataclass
class RuntimeCfg:
    ignition: IgnitionCfg = field(default_factory=IgnitionCfg)
    telemetry: TelemetryCfg = field(default_factory=TelemetryCfg)
    replay: ReplayCfg = field(default_factory=ReplayCfg)


def load_runtime_cfg(path: str | Path = "config/runtime.yaml") -> RuntimeCfg:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return RuntimeCfg()
    try:
        payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return RuntimeCfg()
    ignition = _merge_dataclass(IgnitionCfg(), payload.get("ignition", {}))
    telemetry = _merge_dataclass(TelemetryCfg(), payload.get("telemetry", {}))
    replay = _merge_dataclass(ReplayCfg(), payload.get("replay", {}))
    return RuntimeCfg(ignition=ignition, telemetry=telemetry, replay=replay)


def _merge_dataclass(instance, overrides: dict[str, Any]):
    data = instance.__dict__.copy()
    for key, value in (overrides or {}).items():
        if key in data:
            data[key] = value
    return instance.__class__(**data)


__all__ = ["load_runtime_cfg", "RuntimeCfg", "IgnitionCfg", "TelemetryCfg", "ReplayCfg"]
