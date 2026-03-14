from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Tuple
import time

import numpy as np

from eqnet_core.models.emotion import ValueGradient


@dataclass
class HeartbeatConfig:
    base_rate: float = 0.85
    gain: float = 0.45
    min_rate: float = 0.2
    max_rate: float = 3.0
    max_dt: float = 2.0


@dataclass
class HeartbeatState:
    rate: float = 0.85
    phase: float = 0.0
    last_ts: Optional[float] = None


class HeartbeatCore:
    def __init__(
        self,
        config: Optional[HeartbeatConfig] = None,
        state: Optional[HeartbeatState] = None,
    ) -> None:
        self.config = config or HeartbeatConfig()
        self.state = state or HeartbeatState(
            rate=self.config.base_rate,
            phase=0.0,
            last_ts=time.time(),
        )

    def set_params(
        self,
        *,
        base_rate: Optional[float] = None,
        gain: Optional[float] = None,
    ) -> None:
        if base_rate is not None:
            self.config.base_rate = float(base_rate)
        if gain is not None:
            self.config.gain = float(gain)

    def update(
        self,
        arousal: float,
        *,
        emotion_step: Callable[..., Any],
        noise_scale: Optional[float] = None,
        damp: float = 0.02,
    ) -> Tuple[float, float, np.ndarray]:
        now = time.time()
        dt = 0.0
        if self.state.last_ts is not None:
            dt = max(now - self.state.last_ts, 0.0)
        self.state.last_ts = now

        arousal_unit = float(np.clip(arousal, 0.0, 1.0))
        rate = float(
            np.clip(
                self.config.base_rate + self.config.gain * arousal_unit,
                self.config.min_rate,
                self.config.max_rate,
            )
        )
        self.state.rate = rate
        self.state.phase = float(
            (self.state.phase + rate * min(dt, self.config.max_dt)) % 1.0
        )
        noise_val = (
            noise_scale
            if noise_scale is not None
            else max(0.02, 0.01 + 0.02 * abs(arousal))
        )
        phi_vec = emotion_step(
            self.config.base_rate,
            self.config.gain,
            noise_scale=noise_val,
            damp=float(max(damp, 0.0)),
        )
        return self.state.rate, self.state.phase, np.array(phi_vec, copy=True)


class PainStressCore:
    def stress(
        self,
        *,
        sensor_metrics: Mapping[str, Any],
        last_shadow_estimate: Optional[Mapping[str, Any]],
        last_gate_context: Mapping[str, Any],
    ) -> float:
        if isinstance(last_shadow_estimate, Mapping):
            for key in ("stress", "shadow_stress", "risk"):
                if key in last_shadow_estimate:
                    try:
                        return float(last_shadow_estimate[key])
                    except (TypeError, ValueError):
                        pass
        for key in ("future_risk_stress", "tension_score"):
            if key in last_gate_context:
                try:
                    return float(last_gate_context[key])
                except (TypeError, ValueError):
                    pass
        base = float(sensor_metrics.get("body_stress_index", 0.0) or 0.0)
        voice_level = float(sensor_metrics.get("voice_level", 0.0) or 0.0)
        breath_rate = float(sensor_metrics.get("breath_rate", 0.0) or 0.0)
        autonomic_balance = float(sensor_metrics.get("autonomic_balance", 0.5) or 0.5)
        body_state_flag = str(sensor_metrics.get("body_state_flag") or "normal")
        privacy_tags = [str(tag).lower() for tag in (sensor_metrics.get("privacy_tags") or [])]
        pressure = base + 0.12 * voice_level + 0.08 * breath_rate
        if autonomic_balance < 0.42:
            pressure += 0.12
        if body_state_flag == "private_high_arousal" or "private" in privacy_tags:
            pressure += 0.18
        if body_state_flag == "overloaded":
            pressure += 0.1
        return float(np.clip(pressure, 0.0, 1.0))


class RecoveryCore:
    def recovery_need(self, *, stress: float, current_energy: float) -> float:
        return float(np.clip(0.55 * stress + 0.45 * (1.0 - current_energy), 0.0, 1.0))

    def attention_density(
        self,
        *,
        sensor_metrics: Mapping[str, Any],
        last_gate_context: Mapping[str, Any],
    ) -> float:
        if isinstance(last_gate_context.get("life_indicator"), (int, float)):
            return float(
                np.clip(float(last_gate_context.get("life_indicator", 0.0)), 0.0, 1.0)
            )
        activity = float(sensor_metrics.get("activity_level", 0.0) or 0.0)
        voice_level = float(sensor_metrics.get("voice_level", 0.0) or 0.0)
        person_count = min(float(sensor_metrics.get("person_count", 0.0) or 0.0), 3.0) / 3.0
        density = 0.55 * activity + 0.25 * voice_level + 0.2 * person_count
        return float(np.clip(density, 0.0, 1.0))


class BoundaryCore:
    def safety_bias(
        self,
        *,
        value_gradient: ValueGradient,
        safety_lens: Callable[[ValueGradient], float],
    ) -> float:
        return float(safety_lens(value_gradient))

