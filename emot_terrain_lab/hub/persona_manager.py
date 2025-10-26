# -*- coding: utf-8 -*-
"""Persona orchestration with Bayesian mode selection."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from persona import compose_controls as persona_compose_controls

from ..persona.selector_bayes import BayesPersonaSelector


class PersonaManager:
    """Apply persona configuration and manage Bayesian mode adaptation."""

    def __init__(
        self,
        *,
        persona_cfg: Optional[Dict[str, object]],
        heartiness: float,
        bayes_selector: Optional[BayesPersonaSelector] = None,
    ) -> None:
        self._persona_cfg = persona_cfg or {}
        self._heartiness = float(heartiness)
        self._selector = bayes_selector

    def prepare(
        self,
        ctx: Dict[str, object],
        base_controls: Dict[str, float],
    ) -> Tuple[Dict[str, float], Optional[Dict[str, object]], Optional[str]]:
        """Return updated controls, persona payload, and chosen mode."""
        persona_cfg = dict(self._persona_cfg) if self._persona_cfg else {}
        persona_enabled = persona_cfg.get("enabled", True)
        chosen_mode: Optional[str] = persona_cfg.get("mode")

        if persona_enabled and self._selector is not None and not ctx.get("mode_style"):
            try:
                chosen_mode = self._selector.choose()
                persona_cfg["mode"] = chosen_mode
            except Exception:
                chosen_mode = persona_cfg.get("mode")

        if not persona_cfg:
            return base_controls, None, chosen_mode

        try:
            persona_result = persona_compose_controls(
                culture_name=str(persona_cfg.get("culture", "anime_2010s_slice")),
                mode_name=str(persona_cfg.get("mode", "caregiver")),
                user_pref=persona_cfg.get("user_pref", {}),
                alpha=float(persona_cfg.get("alpha", self._heartiness)),
                beta=float(persona_cfg.get("beta", 0.3)),
                safety=persona_cfg.get("safety", {}),
                base_controls=base_controls,
            )
            controls = persona_result.controls
            return controls, persona_result.to_dict(), chosen_mode
        except Exception:
            return base_controls, None, chosen_mode

    def update(self, mode: Optional[str], success: bool, d_tau: float) -> None:
        if self._selector is None or not mode:
            return
        try:
            self._selector.update(mode, success=success, d_tau=d_tau)
        except Exception:
            pass

    def metrics(self) -> Dict[str, Dict[str, float]]:
        if self._selector is None:
            return {}
        return self._selector.metrics()

    def set_forgetting_params(self, params: Optional[Dict[str, float]]) -> None:
        if self._selector is None or not params:
            return
        halflife = params.get("halflife_tau")
        if halflife is None:
            return
        try:
            self._selector.set_halflife_tau(float(halflife))
        except Exception:
            pass


__all__ = ["PersonaManager"]
