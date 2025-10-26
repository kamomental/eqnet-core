# -*- coding: utf-8 -*-
"""Replay orchestration helper."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..mind.replay_unified import UnifiedReplayKernel
from ..mind.self_model import SelfModel
from ..timekeeper import TimeKeeper
from ..utils.io import append_jsonl
from .phase_ratio_filter import HysteresisCfg, PhaseRatioFilter
from devlife.mind.replay_memory import ReplayMemory


class ReplayExecutor:
    """Encapsulate replay triggering, execution, and logging."""

    def __init__(
        self,
        *,
        mode: str,
        heartiness: float,
        replay_cfg: Dict[str, Any],
        replay_kernel: UnifiedReplayKernel,
        replay_memory: ReplayMemory,
        self_model: SelfModel,
        timekeeper: TimeKeeper,
        value_weights: Dict[str, float],
        logs_cfg: Optional[Dict[str, Any]] = None,
        step_limits: Tuple[int, int] = (1, 6),
    ) -> None:
        self.mode = mode.lower()
        self.heartiness = float(heartiness)
        self.replay_cfg = replay_cfg
        self.replay_kernel = replay_kernel
        self.replay_memory = replay_memory
        self.self_model = self_model
        self.timekeeper = timekeeper
        self.value_weights = value_weights
        logs_cfg = logs_cfg or {}
        self.log_path = logs_cfg.get("replay_firings_path", "logs/replay_firings.jsonl")
        self.step_limits = (max(1, int(step_limits[0])), max(1, int(step_limits[1])))
        self._horizon_override: Optional[int] = None
        self._min_steps = self.step_limits[0]
        phase_cfg = self.replay_cfg.get("phase_transition", {}) or {}
        self._phase_cfg = {
            "reverse_max": float(phase_cfg.get("reverse_max", 0.8)),
            "reverse_min": float(phase_cfg.get("reverse_min", 0.2)),
            "success_streak": int(phase_cfg.get("success_streak", 6)),
            "error_alpha": float(phase_cfg.get("error_alpha", 0.2)),
            "error_target": float(phase_cfg.get("error_target", 0.2)),
            "success_threshold": float(phase_cfg.get("success_threshold", 0.05)),
        }
        self._phase_success_streak = 0
        self._phase_error = float(self._phase_cfg["error_target"])

        phase_dyn_cfg = self.replay_cfg.get("phase", {}) or {}
        self._phase_filter: PhaseRatioFilter | None = None
        if phase_dyn_cfg:
            hysteresis_cfg = phase_dyn_cfg.get("hysteresis", {}) or {}
            self._phase_filter = PhaseRatioFilter(
                HysteresisCfg(
                    ema_tau=float(phase_dyn_cfg.get("ema_tau", 6.0)),
                    hysteresis_up=float(hysteresis_cfg.get("up", hysteresis_cfg.get("reverse", 0.6))),
                    hysteresis_down=float(hysteresis_cfg.get("down", hysteresis_cfg.get("forward", 0.4))),
                    min_window_tau=float(phase_dyn_cfg.get("min_window_tau", 0.0)),
                    consecutive_up=int(hysteresis_cfg.get("consecutive_up", 1)),
                    consecutive_down=int(hysteresis_cfg.get("consecutive_down", 1)),
                )
            )

    def _should_trigger(self, ctx: Dict[str, Any], cooldown_tau: float) -> bool:
        signals = ctx
        trigger_signals = (
            signals.get("uncertainty", 0.3) > 0.5
            or signals.get("novelty", 0.2) > 0.5
            or abs(signals.get("dU_est", 0.0)) > 0.2
            or signals.get("norm_risk", 0.0) > 0.4
            or signals.get("pressure", 0.0) > 0.5
        )
        cooled_down = self.timekeeper.since_last("replay") >= cooldown_tau
        return bool(trigger_signals and cooled_down)

    def run(
        self,
        ctx_time: Dict[str, Any],
        plan: Dict[str, Any],
        norms: Dict[str, Any],
        safety_ctx: Dict[str, Any],
        tau_rate: float,
    ) -> Dict[str, Any]:
        replay_info: Dict[str, Any] = {"fired": False, "budget": 0, "horizon": 0}
        replay_details: Optional[Dict[str, Any]] = None
        best_choice: Optional[Dict[str, Any]] = None
        candidates: List[Dict[str, Any]] = []
        d_tau_step: Optional[float] = None

        if self.mode not in {"reflective", "living"}:
            return {
                "info": replay_info,
                "details": replay_details,
                "best": best_choice,
                "candidates": candidates,
                "d_tau_step": d_tau_step,
            }

        cooldown_tau = float(self.replay_cfg.get("cooldown_tau", 0.8))
        if not self._should_trigger(ctx_time, cooldown_tau):
            return {
                "info": replay_info,
                "details": replay_details,
                "best": best_choice,
                "candidates": candidates,
                "d_tau_step": d_tau_step,
            }

        field_signals = plan.get("biofield", {}).get("signals", {}) if isinstance(
            plan.get("biofield"), dict) else {}
        h_base = float(self.replay_cfg.get("h_base_tau", 2.0))
        horizon_tau = _clip(h_base * (0.7 + 0.6 * self.heartiness) * tau_rate, 0.8, 6.0)
        step_map = self.replay_cfg.get("d_tau_step", {"dialogue": 0.7, "maze": 1.0, "motor": 0.5})
        adapter_name = self.replay_cfg.get("adapter", "dialogue")
        d_tau_step = float(step_map.get(adapter_name, 0.7))
        step_lo, step_hi = self.step_limits
        step_lo = max(step_lo, getattr(self, "_min_steps", step_lo))
        raw_steps = int(math.ceil(horizon_tau / max(0.1, d_tau_step)))
        steps = int(_clip(raw_steps, step_lo, step_hi))
        limits_trace = {"raw": raw_steps, "base": steps}
        if self._horizon_override is not None:
            after_override = min(steps, max(step_lo, self._horizon_override))
            limits_trace["forgetting_cap"] = after_override
            steps = after_override

        meta_ctx = ctx_time.get("metamemory", {}) if isinstance(ctx_time, Mapping) else {}
        if meta_ctx.get("tot_active"):
            tot_cap = int(meta_ctx.get("horizon_cap", 2))
            tot_cap = max(step_lo, min(step_hi, tot_cap))
            if steps > tot_cap:
                limits_trace["tot_cap"] = tot_cap
                steps = tot_cap
        inflammation = float(field_signals.get("inflammation_global", 0.0))
        if inflammation > 0.0:
            after_inflammation = max(step_lo, int(round(steps * (1.0 - 0.5 * inflammation))))
            limits_trace["after_inflammation"] = after_inflammation
            steps = after_inflammation
        steps_bias = float(plan.get("controls", {}).get("steps_bias", 0.0)) if isinstance(
            plan.get("controls"), dict) else 0.0
        if steps_bias != 0.0:
            after_bias = int(_clip(steps + steps_bias, step_lo, step_hi))
            limits_trace["after_bias"] = after_bias
            steps = after_bias
        if safety_ctx.get("bayes_decision") in {"READ_ONLY", "BLOCK"} or safety_ctx.get("read_only"):
            capped = min(step_hi, max(step_lo, min(steps, 2)))
            limits_trace["safety_cap"] = capped
            steps = capped
        limits_trace["applied"] = steps
        seeds = self.replay_memory.sample_prioritized(
            k=int(self.replay_cfg.get("seed_topk", 3)),
            limit=int(self.replay_cfg.get("seed_limit", 200)),
        )
        coherence_baseline = self.self_model.coherence()

        reverse_ratio = self._compute_reverse_ratio()
        best_choice, candidates = self.replay_kernel.run(
            z0=ctx_time.get("latent", {}),
            steps=steps,
            d_tau_step=d_tau_step,
            tau_rate=tau_rate,
            mood=plan["mood"],
            norms=norms,
            weights=self.value_weights,
            read_only_policy=safety_ctx["read_only"],
            cand_cap=int(self.replay_cfg.get("cand_cap", 64)),
            kappa=float(self.replay_cfg.get("kappa", 0.25)),
            coherence_baseline=coherence_baseline,
            coherence_cb=self.self_model.preview_coherence,
            reverse_ratio=reverse_ratio,
        )
        self.timekeeper.mark("replay")
        replay_info = {"fired": True, "budget": steps, "horizon": horizon_tau}

        best_action = best_choice.get("a") if best_choice else None
        if best_action is not None:
            try:
                self.self_model.note(
                    {
                        "id": len(self.self_model.narrative.events),
                        "intent": str(best_action),
                        "cause": None,
                        "coherence": best_choice.get("coherence"),
                    }
                )
            except Exception:
                pass

        replay_details = {
            "type": "forward",
            "horizon_tau": round(horizon_tau, 2),
            "steps": steps,
            "best_action": best_choice.get("a") if best_choice else None,
            "utility": round(best_choice.get("U", 0.0) if best_choice else 0.0, 3),
            "policy_update": not (best_choice.get("read_only", False) if best_choice else False),
            "coherence": round(float(best_choice.get("coherence", coherence_baseline)), 3)
            if best_choice
            else round(float(coherence_baseline), 3),
            "candidates": candidates[:10],
            "seeds": seeds,
            "value_summary": best_choice.get("summary", {}) if best_choice else {},
            "value_weights": dict(self.value_weights),
            "bayes_safety": {
                "category": safety_ctx["act_category"],
                "decision": safety_ctx["bayes_decision"]
                or ("READ_ONLY" if safety_ctx["read_only"] else "ALLOW"),
            },
            "limits": limits_trace,
            "reverse_ratio": round(reverse_ratio, 3),
        }

        self._record_log(replay_details, candidates)
        success = bool(
            best_choice
            and float(best_choice.get("U", 0.0)) >= self._phase_cfg["success_threshold"]
            and not safety_ctx.get("read_only", False)
            and not ctx_time.get("misfire", False)
        )
        error_signal = max(0.0, -float(best_choice.get("U", 0.0)) if best_choice else 0.0)
        self._update_phase(success, error_signal)

        return {
            "info": replay_info,
            "details": replay_details,
            "best": best_choice,
            "candidates": candidates,
            "d_tau_step": d_tau_step,
        }

    def set_forgetting_params(self, params: Optional[Dict[str, Any]]) -> None:
        """Apply forgetting controller inputs for replay horizon."""
        if not params:
            self._horizon_override = None
            return
        horizon = params.get("horizon")
        if horizon is None:
            return
        try:
            self._horizon_override = max(1, int(horizon))
        except Exception:
            self._horizon_override = None

    def set_min_steps(self, min_steps: Optional[int]) -> None:
        if min_steps is None:
            self._min_steps = self.step_limits[0]
            return
        try:
            self._min_steps = max(self.step_limits[0], int(min_steps))
        except Exception:
            self._min_steps = self.step_limits[0]

    def _record_log(self, replay_details: Optional[Dict[str, Any]], candidates: List[Dict[str, Any]]) -> None:
        if not replay_details:
            return
        payload = {
            "ts": replay_details.get("ts"),
            "U_top": max((float(c.get("U", 0.0)) for c in candidates), default=0.0),
            "steps": replay_details.get("steps"),
            "horizon_tau": replay_details.get("horizon_tau"),
            "coherence": replay_details.get("coherence"),
            "safety_decision": replay_details.get("bayes_safety", {}).get("decision"),
        }
        if payload["ts"] is None:
            from time import time as _time

            payload["ts"] = _time()
        replay_details["ts"] = payload["ts"]
        append_jsonl(self.log_path, payload)

    def _current_tau(self) -> float:
        tau_now = getattr(self.timekeeper, "tau_now", None)
        if callable(tau_now):
            try:
                return float(tau_now())
            except Exception:
                pass
        return float(getattr(self.timekeeper, "tau", 0.0))

    def _compute_reverse_ratio(self) -> float:
        success_streak_target = max(1, self._phase_cfg["success_streak"])
        phase_success = min(1.0, self._phase_success_streak / success_streak_target)
        error_target = max(1e-6, self._phase_cfg["error_target"])
        phase_error = max(0.0, min(1.0, 1.0 - self._phase_error / error_target))
        phase = 0.5 * phase_success + 0.5 * phase_error
        reverse_max = self._phase_cfg["reverse_max"]
        reverse_min = self._phase_cfg["reverse_min"]
        ratio = float(reverse_max * (1.0 - phase) + reverse_min * phase)
        if self._phase_filter is not None:
            ratio = float(self._phase_filter.update(ratio, self._current_tau()))
        return ratio

    def _update_phase(self, success: bool, error_signal: float) -> None:
        if success:
            self._phase_success_streak += 1
        else:
            self._phase_success_streak = 0
        alpha = self._phase_cfg["error_alpha"]
        self._phase_error = float((1.0 - alpha) * self._phase_error + alpha * max(0.0, error_signal))


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


__all__ = ["ReplayExecutor"]
