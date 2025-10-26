# -*- coding: utf-8 -*-
"""Autopilot supervisor for on-turn safety, hygiene, and stability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class AutopilotCfg:
    assoc_th: float = 0.15
    nat_th: float = 0.25
    r_max: float = 0.78
    latency_p95_ceiling: float = 1.10
    msv_k_min: int = 2
    heartiness_bounds: tuple[float, float] = (0.1, 0.9)
    gain_bounds: tuple[float, float] = (0.0, 0.10)


class Autopilot:
    """Coordinate safety/stability knobs with minimal intervention."""

    def __init__(
        self,
        cfg: AutopilotCfg,
        *,
        safety_gate: Any | None,
        forgetting: Any | None,
        solver_mgr: Any | None,
        thought_bus: Any | None,
        heartiness_start: float = 0.4,
    ) -> None:
        self.cfg = cfg
        self.safety_gate = safety_gate
        self.forgetting = forgetting
        self.solver = solver_mgr
        self.thought_bus = thought_bus
        lo, hi = cfg.heartiness_bounds
        self.state: Dict[str, float] = {
            "heartiness": max(lo, min(hi, heartiness_start)),
            "tb_gain_cap": cfg.gain_bounds[1],
        }

    # --------------------------------------------------------------------- #
    # Turn lifecycle hooks

    def pre_plan(self, metrics: Dict[str, float], ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Called before planning; prime safety posture and TB cap."""
        reason = "hold"
        mode = ctx.get("mode_default")
        if self.safety_gate is not None:
            try:
                decision = self.safety_gate.decide(
                    ctx.get("risk_p", 0.0),
                    ctx.get("domain", str(ctx.get("act_category", ""))),
                    bool(ctx.get("consent", True)),
                )
                mode = decision.get("mode", mode)
                reason = decision.get("reason", reason)
            except Exception:
                reason = "safety_gate_error"

        if (
            metrics.get("assoc_defect", 0.0) > self.cfg.assoc_th
            or metrics.get("naturality_residual", 0.0) > self.cfg.nat_th
        ):
            if hasattr(self.solver, "auto_upshift"):
                try:
                    self.solver.auto_upshift()
                except Exception:
                    pass

        if metrics.get("r", 0.0) > self.cfg.r_max:
            self.state["tb_gain_cap"] = 0.0
        else:
            self.state["tb_gain_cap"] = self.cfg.gain_bounds[1]

        return {
            "mode": mode,
            "msv_k_min": int(self.cfg.msv_k_min),
            "tb_gain_cap": float(self.state["tb_gain_cap"]),
            "reason": reason,
        }

    def mid_adjust(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Called mid-turn; tweak heartiness / forgetting bias."""
        jerk = float(metrics.get("jerk_rate", 0.0))
        delta = -0.1 * (jerk - 0.05)
        lo, hi = self.cfg.heartiness_bounds
        heartiness = max(lo, min(hi, self.state["heartiness"] + delta))
        self.state["heartiness"] = heartiness

        if hasattr(self.forgetting, "set_bias"):
            try:
                self.forgetting.set_bias(jerk_bias=jerk)
            except Exception:
                pass

        return {"heartiness": heartiness}

    def post_turn(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Called at turn end; adjust solver load & TB gain cap."""
        if metrics.get("latency_p95_ratio", 1.0) > self.cfg.latency_p95_ceiling:
            if hasattr(self.solver, "auto_downshift"):
                try:
                    self.solver.auto_downshift()
                except Exception:
                    pass

        if hasattr(self.thought_bus, "set_gain_cap"):
            try:
                self.thought_bus.set_gain_cap(self.state["tb_gain_cap"])
            except Exception:
                pass

        return {"tb_gain_cap": float(self.state["tb_gain_cap"])}


__all__ = ["Autopilot", "AutopilotCfg"]

