"""AKOrN gate (minimal): small-gain + soft barrier modulation.

Inputs (metrics):
- R: Kuramoto synchrony [0,1]
- rho: field density/coupling (0+)
- I: Ignition-Index (real)
- q: queue/urgency [0,1] (optional)

Outputs (controls adjusted):
- temperature, top_p, pause_ms (kept within safe bounds)

If metrics are missing, passes controls through unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import math
import os


@dataclass
class AkornConfig:
    # Small gains
    temp_gain_R: float = -0.10
    temp_gain_I: float = -0.08
    temp_gain_rho: float = -0.05

    top_p_gain_R: float = -0.06
    top_p_gain_I: float = -0.05

    pause_gain_I: float = 120.0   # ms per I unit
    pause_gain_R: float = 160.0   # ms per (R-0.5)

    # Soft barrier gains (push back into safe set near boundaries)
    barrier_k_temp: float = 0.08
    barrier_k_top_p: float = 0.06
    barrier_k_pause: float = 80.0

    # Bounds
    temp_min: float = 0.20
    temp_max: float = 0.95
    top_p_min: float = 0.30
    top_p_max: float = 0.98
    pause_min: float = 180.0
    pause_max: float = 1200.0


class AkornGate:
    def __init__(self, config: AkornConfig | None = None) -> None:
        self.cfg = config or AkornConfig()

    @staticmethod
    def from_env() -> "AkornConfig":
        """Build config from environment overrides.

        Supported variables (floats):
        - AKORN_TEMP_GAIN_R, AKORN_TEMP_GAIN_I, AKORN_TEMP_GAIN_RHO
        - AKORN_TOP_P_GAIN_R, AKORN_TOP_P_GAIN_I
        - AKORN_PAUSE_GAIN_I, AKORN_PAUSE_GAIN_R
        - AKORN_TEMP_MIN/MAX, AKORN_TOP_P_MIN/MAX, AKORN_PAUSE_MIN/MAX
        """
        def f(name: str, default: float) -> float:
            val = os.getenv(name)
            if val is None:
                return default
            try:
                return float(val)
            except Exception:
                return default

        cfg = AkornConfig(
            temp_gain_R=f("AKORN_TEMP_GAIN_R", AkornConfig.temp_gain_R),
            temp_gain_I=f("AKORN_TEMP_GAIN_I", AkornConfig.temp_gain_I),
            temp_gain_rho=f("AKORN_TEMP_GAIN_RHO", AkornConfig.temp_gain_rho),
            top_p_gain_R=f("AKORN_TOP_P_GAIN_R", AkornConfig.top_p_gain_R),
            top_p_gain_I=f("AKORN_TOP_P_GAIN_I", AkornConfig.top_p_gain_I),
            pause_gain_I=f("AKORN_PAUSE_GAIN_I", AkornConfig.pause_gain_I),
            pause_gain_R=f("AKORN_PAUSE_GAIN_R", AkornConfig.pause_gain_R),
            barrier_k_temp=f("AKORN_BARRIER_K_TEMP", AkornConfig.barrier_k_temp),
            barrier_k_top_p=f("AKORN_BARRIER_K_TOP_P", AkornConfig.barrier_k_top_p),
            barrier_k_pause=f("AKORN_BARRIER_K_PAUSE", AkornConfig.barrier_k_pause),
            temp_min=f("AKORN_TEMP_MIN", AkornConfig.temp_min),
            temp_max=f("AKORN_TEMP_MAX", AkornConfig.temp_max),
            top_p_min=f("AKORN_TOP_P_MIN", AkornConfig.top_p_min),
            top_p_max=f("AKORN_TOP_P_MAX", AkornConfig.top_p_max),
            pause_min=f("AKORN_PAUSE_MIN", AkornConfig.pause_min),
            pause_max=f("AKORN_PAUSE_MAX", AkornConfig.pause_max),
        )
        return cfg

    def _extract_metrics(self, controls: Dict[str, float], metrics: Dict[str, float] | None) -> Dict[str, float]:
        m = dict(metrics or {})
        # Also accept flattened or namespaced values in controls
        for key in ("R", "rho", "I", "q"):
            if key not in m and key in controls:
                m[key] = controls[key]
        if "akorn" in controls and isinstance(controls["akorn"], dict):
            for k in ("R", "rho", "I", "q"):
                if k not in m and k in controls["akorn"]:
                    m[k] = controls["akorn"][k]
        return m

    def apply(self, controls: Dict[str, float], metrics: Dict[str, float] | None = None) -> Tuple[Dict[str, float], Dict[str, float]]:
        cfg = self.cfg
        out = dict(controls)
        log: Dict[str, float] = {}
        m = self._extract_metrics(controls, metrics)
        if not m:
            return out, log

        R = float(max(0.0, min(1.0, m.get("R", 0.5))))
        rho = float(max(0.0, m.get("rho", 0.0)))
        I = float(m.get("I", 0.0))
        q = float(max(0.0, min(1.0, m.get("q", 0.0))))

        # Temperature
        temp0 = float(out.get("temperature", 0.65))
        dtemp = cfg.temp_gain_R * (R - 0.5) + cfg.temp_gain_I * I + cfg.temp_gain_rho * max(0.0, rho - 1.0)
        # queue urgency can boost exploration slightly
        dtemp += 0.05 * (q - 0.5)
        temp1 = temp0 + dtemp
        # Soft barrier to keep within [min,max]
        temp1 += self._soft_barrier_push(temp1, cfg.temp_min, cfg.temp_max, cfg.barrier_k_temp)
        temp1 = float(max(cfg.temp_min, min(cfg.temp_max, temp1)))

        # top_p
        top_p0 = float(out.get("top_p", 0.85))
        dtop = cfg.top_p_gain_R * (R - 0.5) + cfg.top_p_gain_I * I + 0.03 * (q - 0.5)
        top_p1 = top_p0 + dtop
        top_p1 += self._soft_barrier_push(top_p1, cfg.top_p_min, cfg.top_p_max, cfg.barrier_k_top_p)
        top_p1 = float(max(cfg.top_p_min, min(cfg.top_p_max, top_p1)))

        # pause_ms (longer pause when I positive or high R)
        pause0 = float(out.get("pause_ms", 360.0))
        dp = cfg.pause_gain_I * I + cfg.pause_gain_R * (R - 0.5) - 60.0 * (q - 0.5)
        pause1 = pause0 + dp
        pause1 += self._soft_barrier_push(pause1, cfg.pause_min, cfg.pause_max, cfg.barrier_k_pause)
        pause1 = float(max(cfg.pause_min, min(cfg.pause_max, pause1)))

        out["temperature"] = temp1
        out["top_p"] = top_p1
        out["pause_ms"] = pause1

        # Priority proxy from ignition index
        out["priority"] = float(1.0 / (1.0 + math.exp(-I)))  # sigmoid(I)

        log.update({
            "akorn.dtemp": float(temp1 - temp0),
            "akorn.dtop_p": float(top_p1 - top_p0),
            "akorn.dpause_ms": float(pause1 - pause0),
            "akorn.R": R,
            "akorn.rho": rho,
            "akorn.I": I,
            "akorn.q": q,
        })
        return out, log

    @staticmethod
    def _soft_barrier_push(x: float, lo: float, hi: float, k: float) -> float:
        # push inside if outside; near edges add small corrective term
        if x < lo:
            return (lo - x)
        if x > hi:
            return (hi - x)
        # inside: use reciprocal distance to boundaries
        eps = 1e-6
        left = x - lo
        right = hi - x
        return k * (1.0 / (right + eps) - 1.0 / (left + eps)) * 0.0  # neutral unless needed


# Also expose on the config type for convenience
def _akornconfig_from_env_cls(cls: type[AkornConfig]) -> AkornConfig:
    return AkornGate.from_env()


setattr(AkornConfig, "from_env", classmethod(_akornconfig_from_env_cls))


__all__ = ["AkornGate", "AkornConfig"]
