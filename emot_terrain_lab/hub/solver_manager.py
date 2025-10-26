# -*- coding: utf-8 -*-
"""Continuous solver manager for EQNet hybrid modules."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


class SolverManager:
    """Manage integration schemes and substeps for continuous-time modules."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        cfg = dict(config or {})
        self.auto_cfg = cfg.get("auto", {}) or {}
        self.modules: Dict[str, Dict[str, Any]] = {}
        self.max_substeps = int(self.auto_cfg.get("max_substeps", 8))
        self._init_module("biofield", cfg.get("biofield"))
        self._init_module("kuramoto", cfg.get("kuramoto"))
        self._init_module("ssm", cfg.get("ssm"))

    def _init_module(self, name: str, module_cfg: Optional[Mapping[str, Any]]) -> None:
        if module_cfg is None:
            module_cfg = {}
        default = {
            "scheme": "explicit",
            "substeps": "auto",
            "current_substeps": 1,
            "cfl": 0.0,
        }
        merged = dict(default)
        merged.update(module_cfg or {})
        substeps_cfg = merged.get("substeps", "auto")
        if isinstance(substeps_cfg, str) and substeps_cfg.lower() == "auto":
            merged["substeps"] = "auto"
            merged["current_substeps"] = 1
        else:
            try:
                merged["current_substeps"] = max(1, int(substeps_cfg))
            except Exception:
                merged["current_substeps"] = 1
            merged["substeps"] = merged["current_substeps"]
        merged["max_substeps"] = int(merged.get("max_substeps", self.max_substeps))
        self.modules[name] = merged

    def adjust(self, composition: Mapping[str, Any], naturality: Mapping[str, Any]) -> None:
        """Adjust substeps based on recent metrics."""
        assoc_defect = composition.get("assoc_defect")
        residual = naturality.get("residual")
        assoc_th = float(self.auto_cfg.get("assoc_defect_threshold", 0.15))
        resid_th = float(self.auto_cfg.get("residual_threshold", 0.25))
        factor = max(2, int(self.auto_cfg.get("substep_factor", 2)))
        relax_factor = max(2, int(self.auto_cfg.get("substep_relax_factor", factor)))

        high = ((assoc_defect is not None and assoc_defect > assoc_th) or
                (residual is not None and residual > resid_th))
        low = (
            assoc_defect is not None
            and assoc_defect < assoc_th * 0.5
            and residual is not None
            and residual < resid_th * 0.5
        )

        for module in self.modules.values():
            if module.get("substeps") != "auto":
                continue
            current = max(1, int(module.get("current_substeps", 1)))
            max_allowed = max(1, int(module.get("max_substeps", self.max_substeps)))
            if high:
                module["current_substeps"] = min(current * factor, max_allowed)
            elif low and current > 1:
                module["current_substeps"] = max(1, current // relax_factor)

    def step_biofield(self, bus, total_dt: float) -> None:
        """Integrate InteroBus with current solver settings."""
        if bus is None:
            return
        module = self.modules.get("biofield")
        if module is None:
            bus.step(total_dt)
            return
        substeps = max(1, int(module.get("current_substeps", 1)))
        if substeps > 64:
            substeps = 64
            module["current_substeps"] = substeps
        dt = total_dt / substeps if substeps > 0 else total_dt
        for _ in range(substeps):
            bus.step(dt)
        alpha = getattr(bus, "alpha", None)
        if alpha is not None:
            module["cfl"] = float(alpha * dt)
        module["last_dt"] = float(total_dt)

    def auto_upshift(self, factor: int | None = None) -> None:
        """Increase auto substeps to stabilise integration."""
        factor = max(2, int(factor or self.auto_cfg.get("substep_factor", 2)))
        for module in self.modules.values():
            if module.get("substeps") != "auto":
                continue
            cur = max(1, int(module.get("current_substeps", 1)))
            max_allowed = max(1, int(module.get("max_substeps", self.max_substeps)))
            module["current_substeps"] = min(cur * factor, max_allowed)

    def auto_downshift(self, relax_factor: int | None = None) -> None:
        """Decrease auto substeps to relieve compute pressure."""
        relax_factor = max(2, int(relax_factor or self.auto_cfg.get("substep_relax_factor", 2)))
        for module in self.modules.values():
            if module.get("substeps") != "auto":
                continue
            cur = max(1, int(module.get("current_substeps", 1)))
            if cur > 1:
                module["current_substeps"] = max(1, cur // relax_factor)

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Return a serialisable snapshot for receipts/telemetry."""
        snap: Dict[str, Dict[str, Any]] = {}
        for name, module in self.modules.items():
            snap[name] = {
                "scheme": module.get("scheme"),
                "substeps": int(module.get("current_substeps", 1)),
            }
            if "cfl" in module:
                snap[name]["cfl"] = round(float(module.get("cfl", 0.0)), 4)
            if "last_dt" in module:
                snap[name]["last_dt"] = round(float(module.get("last_dt", 0.0)), 4)
        return snap


__all__ = ["SolverManager"]
