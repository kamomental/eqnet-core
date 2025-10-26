# -*- coding: utf-8 -*-
"""Generic fast-path summarizer for multiple task profiles."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Sequence

from ..utils.final_functor import AggregatorKind, colim_over_index, is_cofinal_subset_poset
from ..mind.inner_replay import InnerReplayController, ReplayConfig, ReplayInputs, ReplayOutcome
from ..utils.hashcfg import cfg_fingerprint
from .task_profiles import TaskProfile


def summarize_task_fastpath(
    profile: TaskProfile,
    J_index: Sequence[float],
    projector_map: Mapping[str, Callable[[float], Any]],
    *,
    include_full_labels: bool = True,
) -> Dict[str, Any]:
    """Aggregate cocontinuous features using the profile's checkpoint chain."""

    I = profile.checkpoints
    final_ok = bool(I) and is_cofinal_subset_poset(I, J_index)

    fast: Dict[str, Any] = {}
    needs_full = []

    for name, agg_kind in profile.features_cocont.items():
        projector = projector_map.get(name)
        if projector is None:
            continue
        if final_ok:
            fast[name] = colim_over_index(agg_kind, I, projector)
        else:
            needs_full.append(name)

    needs_full.extend(profile.features_non_cocont)

    dedup_full: list[str] = []
    seen: set[str] = set()
    for item in needs_full:
        if item in seen:
            continue
        seen.add(item)
        dedup_full.append(item)

    predicates: Dict[str, Any] = {}
    if final_ok and profile.fast_predicates:
        for key, spec in profile.fast_predicates.items():
            predicates[key] = _evaluate_predicate(spec, I, projector_map)

    labels = None
    if include_full_labels:
        labels = {
            "fast_features": list(profile.features_cocont.keys()),
            "full_features": list(profile.features_non_cocont),
        }

    return {
        "profile": profile.name,
        "final_ok": final_ok,
        "fast": fast,
        "needs_full": dedup_full,
        "predicates": predicates,
        "labels": labels,
    }


def _evaluate_predicate(
    spec: Mapping[str, Any],
    checkpoints: Sequence[float],
    projector_map: Mapping[str, Callable[[float], Any]],
) -> Any:
    """Evaluate fast predicates that can ride on the checkpoint chain (OR semantics)."""

    predicate_type = spec.get("type")
    if predicate_type == "go_sc_and_rarity":
        pmin = float(spec.get("pmin", 0.9))
        rmin = float(spec.get("rmin", 0.8))
        proj_p = projector_map.get("go_percentile_stream")
        proj_r = projector_map.get("rarity_stream")
        if not proj_p or not proj_r:
            return False
        for t in checkpoints:
            try:
                if float(proj_p(t)) >= pmin and float(proj_r(t)) >= rmin:
                    return True
            except Exception:
                continue
        return False

    # Unknown predicate → False by default
    return False


DEFAULT_REPLAY_CONFIG = ReplayConfig(
    theta_prep=0.7,
    tau_conscious_s=0.32,
    steps=16,
    w_reward=1.0,
    w_risk=0.6,
    w_daff=0.4,
    w_uncert=0.3,
    w_tom_cost=0.25,
    tau_execute=0.0,
    beta_veto=1.0,
    seed=227,
    keep_s_trace=False,
    trace_keep_prob=0.05,
)


def decide_fastpath(
    context: Mapping[str, Any],
    *,
    controller: InnerReplayController | None = None,
) -> Dict[str, Any]:
    """
    Fast-path 最終決定のヒントとして内的リプレイ評価を実行し、receipt を返す。

    実際の action は既存 fast-path ポリシー優先。`context["fallback_decision"]` が
    与えられた場合はそれを final に採用し、inner replay の decision は記録のみ。
    """

    ctrl = controller or InnerReplayController(DEFAULT_REPLAY_CONFIG)
    cfg = ctrl.config
    inputs = ReplayInputs(
        chaos_sens=float(context.get("chaos_sens", 0.3)),
        tom_cost=float(context.get("tom_cost", 0.3)),
        delta_aff_abs=float(context.get("delta_aff_abs", 0.3)),
        risk=float(context.get("risk", 0.3)),
        uncertainty=float(context.get("uncertainty", 0.3)),
        reward_estimate=float(context.get("reward_estimate", 0.7)),
        mood_valence=float(context.get("mood_valence", 0.0)),
        mood_arousal=float(context.get("mood_arousal", 0.0)),
    )
    outcome = ctrl.run_cycle(inputs)
    receipt: Dict[str, Any] = {"decisions": []}
    receipt["inner_replay"] = _serialize_outcome(outcome, cfg)
    receipt["decisions"].append({"fastpath_inner_replay": outcome.decision})
    final = context.get("fallback_decision")
    receipt["final"] = final or outcome.decision
    return receipt


def _serialize_outcome(outcome: ReplayOutcome, cfg: ReplayConfig) -> Dict[str, Any]:
    return {
        "prep": outcome.prep_features,
        "plan": outcome.plan_features,
        "veto": {"score": outcome.veto_score, "decision": outcome.decision},
        "felt_intent_time": outcome.felt_intent_time,
        "u_hat": outcome.u_hat,
        "meta": {
            "cfg_b2": cfg_fingerprint(cfg),
            "seed": cfg.seed,
            "steps": cfg.steps,
            "schema": 1,
        },
    }


__all__ = ["summarize_task_fastpath", "decide_fastpath"]
