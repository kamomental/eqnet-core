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
class EmotionCfg:
    valence_w_rho: float = field(default=1.0)
    valence_w_s: float = field(default=1.0)
    resonance_k: float = field(default=0.05)
    affective_log_path: str = field(default="memory/affective_log.jsonl")


@dataclass
class CultureCfg:
    tag: str = field(default="default")
    politeness: float = field(default=0.5)
    intimacy: float = field(default=0.5)
    feedback: "CultureFeedbackCfg" = field(default_factory=lambda: CultureFeedbackCfg())


@dataclass
class AlertsCfg:
    max_abs_valence_mean: float = field(default=0.6)
    min_corr_rho_I: float = field(default=0.2)
    min_corr_arousal_I: float = field(default=0.1)
    min_corr_rho_rho: float = field(default=0.2)
    max_allowed_lag: float = field(default=8.0)
    min_resonance_samples: int = field(default=100)
    max_abs_culture_valence_mean: float = field(default=0.6)
    min_culture_rho_mean: float = field(default=0.2)
    min_culture_samples: int = field(default=20)


@dataclass
class ResonanceCfg:
    logs: list[str] = field(default_factory=list)
    resample_ms: float | None = field(default=20.0)
    zscore: bool = field(default=True)
    detrend: bool = field(default=True)
    window: str = field(default="hann")
    alpha: float = field(default=0.0)
    beta: float = field(default=0.0)


@dataclass
class RuntimeCfg:
    ignition: IgnitionCfg = field(default_factory=IgnitionCfg)
    telemetry: TelemetryCfg = field(default_factory=TelemetryCfg)
    replay: ReplayCfg = field(default_factory=ReplayCfg)
    emotion: EmotionCfg = field(default_factory=EmotionCfg)
    culture: CultureCfg = field(default_factory=CultureCfg)
    alerts: AlertsCfg = field(default_factory=AlertsCfg)
    resonance: ResonanceCfg = field(default_factory=ResonanceCfg)


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
    emotion = _merge_dataclass(EmotionCfg(), payload.get("emotion", {}))
    culture = _merge_dataclass(
        CultureCfg(),
        payload.get("culture", {}),
        extra_factories={"feedback": CultureFeedbackCfg},
    )
    alerts = _merge_dataclass(AlertsCfg(), payload.get("alerts", {}))
    resonance = _merge_dataclass(ResonanceCfg(), payload.get("resonance", {}))
    return RuntimeCfg(
        ignition=ignition,
        telemetry=telemetry,
        replay=replay,
        emotion=emotion,
        culture=culture,
        alerts=alerts,
        resonance=resonance,
    )


def _merge_dataclass(instance, overrides: dict[str, Any], extra_factories: dict[str, Any] | None = None):
    data = instance.__dict__.copy()
    for key, value in (overrides or {}).items():
        if key not in data:
            continue
        if extra_factories and key in extra_factories and isinstance(value, dict):
            factory_cls = extra_factories[key]
            data[key] = _merge_dataclass(factory_cls(), value)
        else:
            data[key] = value
    return instance.__class__(**data)


@dataclass
class CultureFeedbackCfg:
    enabled: bool = field(default=False)
    corr_high: float = field(default=0.65)
    corr_low: float = field(default=0.35)
    delta: float = field(default=0.02)
    clamp_min: float = field(default=0.0)
    clamp_max: float = field(default=1.0)
    vision_coefficients: dict[str, dict[str, float]] = field(default_factory=dict)


__all__ = [
    "load_runtime_cfg",
    "RuntimeCfg",
    "IgnitionCfg",
    "TelemetryCfg",
    "ReplayCfg",
    "EmotionCfg",
    "CultureCfg",
    "CultureFeedbackCfg",
    "AlertsCfg",
    "ResonanceCfg",
]
