# -*- coding: utf-8 -*-
"""Multi-layer forgetting controller for EQNet."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping


@dataclass
class ForgettingAdvice:
    lstm: Dict[str, float]
    ssm: Dict[str, float]
    intero: Dict[str, float]
    replay: Dict[str, float]
    persona: Dict[str, float]


class ForgettingController:
    """Coordinate forgetting strengths across short/medium/field layers."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enable", True))
        self.lstm_cfg = cfg.get("lstm", {}) or {}
        self.ssm_cfg = cfg.get("ssm", {}) or {}
        self.intero_cfg = cfg.get("intero", {}) or {}
        self.replay_cfg = cfg.get("replay", {}) or {}
        self.persona_cfg = cfg.get("persona", {}) or {}
        self.cfl_max = float(cfg.get("cfl_max", 0.85))

    def advise(
        self,
        *,
        tau_rate: float,
        inflammation: float,
        uncertainty: float,
        novelty: float,
        cfl: float,
    ) -> ForgettingAdvice:
        if not self.enabled:
            return ForgettingAdvice(
                lstm={"forget_bias_delta": 0.0},
                ssm={"stabilize_tau": self.ssm_cfg.get("stabilize_tau", 0.1), "rho_max": self.ssm_cfg.get("rho_max", 0.98)},
                intero={"alpha": self.intero_cfg.get("alpha0", 0.25), "D_scale": 1.0},
                replay={"horizon": max(1, int(self.replay_cfg.get("base_horizon", 2)))},
                persona={"halflife_tau": float(self.persona_cfg.get("halflife_tau", {}).get("base", 24.0))},
            )

        # LSTM: adjust forget bias using tau_rate & inflammation
        alpha_f = float(self.lstm_cfg.get("alpha_f", 0.25))
        beta_f = float(self.lstm_cfg.get("beta_f", 0.20))
        bias_clip = float(self.lstm_cfg.get("bias_clip", 0.40))
        forget_bias_delta = alpha_f * (tau_rate - 1.0) + beta_f * inflammation
        forget_bias_delta = max(-bias_clip, min(bias_clip, forget_bias_delta))

        # SSM: ensure stabilizer and spectral radius clamp
        stabilize_tau = float(self.ssm_cfg.get("stabilize_tau", 0.10))
        rho_max = float(self.ssm_cfg.get("rho_max", 0.98))

        # Intero/Biofield: alpha increases with inflammation; diffusion scaling if CFL high
        alpha0 = float(self.intero_cfg.get("alpha0", 0.25))
        gamma = float(self.intero_cfg.get("gamma", 0.35))
        alpha_eff = alpha0 + gamma * inflammation
        D_scale = 1.0
        if cfl > self.cfl_max:
            D_scale = float(self.intero_cfg.get("D_scale_on_cfl", 0.8))

        # Replay: horizon base +- contributions
        base_h = int(self.replay_cfg.get("base_horizon", 2))
        dh_u_cfg = self.replay_cfg.get("delta_h_uncertainty", {}) or {"k": 2.0, "max": 2}
        dh_i_cfg = self.replay_cfg.get("delta_h_inflammation", {}) or {"k": -2.0, "min": -1}
        dh_u = min(float(dh_u_cfg.get("max", 2)), float(dh_u_cfg.get("k", 2.0)) * uncertainty)
        dh_i = max(float(dh_i_cfg.get("min", -1)), float(dh_i_cfg.get("k", -2.0)) * inflammation)
        horizon = int(max(1, min(6, base_h + dh_u + dh_i)))

        # Persona/Bayes: halflife tau stretched based on inflammation / uncertainty
        persona_half_cfg = self.persona_cfg.get("halflife_tau", {}) or {"base": 24.0, "infl_gain": 0.5}
        half_base = float(persona_half_cfg.get("base", 24.0))
        infl_gain = float(persona_half_cfg.get("infl_gain", 0.5))
        halflife_tau = max(6.0, half_base * (1.0 + infl_gain * inflammation - 0.3 * uncertainty))

        return ForgettingAdvice(
            lstm={"forget_bias_delta": forget_bias_delta},
            ssm={"stabilize_tau": stabilize_tau, "rho_max": rho_max},
            intero={"alpha": alpha_eff, "D_scale": D_scale},
            replay={"horizon": horizon},
            persona={"halflife_tau": halflife_tau},
        )

    def set_bias(self, **kwargs: float) -> None:
        """Accept external hints (currently stored for future use)."""
        if not hasattr(self, "_bias_hints"):
            self._bias_hints: Dict[str, float] = {}
        for key, value in kwargs.items():
            try:
                self._bias_hints[key] = float(value)
            except Exception:
                continue


__all__ = ["ForgettingController", "ForgettingAdvice"]
