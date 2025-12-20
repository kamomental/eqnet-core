"""Access gate with hysteresis and nightly retuning."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class AccessGateConfig:
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    theta: float = 1.0
    ema: float = 0.9
    open_tau: float = 0.6
    close_tau: float = 0.4
    theta_lr: float = 0.1
    target_access: float = 0.2


class AccessGate:
    def __init__(self, config: AccessGateConfig | None = None) -> None:
        self.config = config or AccessGateConfig()
        self.p_ema = 0.0
        self.allow = False

    def decide(
        self,
        u_t: float,
        m_t: float,
        load_t: float,
        override: bool = False,
        reason: str = "normal",
    ) -> dict:
        cfg = self.config
        logit = cfg.alpha * u_t + cfg.gamma * m_t - cfg.beta * load_t - cfg.theta
        p = 1.0 / (1.0 + math.exp(-logit))
        self.p_ema = cfg.ema * self.p_ema + (1.0 - cfg.ema) * p

        if override:
            self.allow = True
            decision_reason = reason or "override"
        else:
            decision_reason = reason
            if not self.allow and self.p_ema >= cfg.open_tau:
                self.allow = True
            elif self.allow and self.p_ema <= cfg.close_tau:
                self.allow = False

        return {
            "u_t": float(u_t),
            "m_t": float(m_t),
            "load_t": float(load_t),
            "logit": float(logit),
            "p_t": float(p),
            "p_ema": float(self.p_ema),
            "allow": self.allow,
            "theta": float(cfg.theta),
            "reason": decision_reason,
        }

    def nightly_retune(self, access_rate: float) -> float:
        cfg = self.config
        cfg.theta += cfg.theta_lr * (access_rate - cfg.target_access)
        return cfg.theta

    def reset(self) -> None:
        self.p_ema = 0.0
        self.allow = False
