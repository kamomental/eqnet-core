#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run a minimal HeartOS loop using InnerReplay as the sole adjudicator."""

from __future__ import annotations

import argparse
import json
import random
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emot_terrain_lab.mind.inner_replay import InnerReplayController, ReplayConfig, ReplayInputs
from emot_terrain_lab.sim.mini_world import MiniWorldScenario, MiniWorldStep, build_default_scenarios
from eqnet_core.models.activation_trace import (
    ActivationNode,
    ActivationTrace,
    ActivationTraceLogger,
    ConfidenceSample,
    ReplayEvent,
)
from eqnet_core.models.emotion import EmotionVector, ValueGradient
from heartos.world_transition import TransitionParams, apply_transition, build_transition_record
from runtime.config import load_runtime_cfg
from telemetry import event as telemetry_event


def _resolve_log_path(template: str, *, day: Optional[datetime] = None) -> Path:
    if "%" not in template:
        return Path(template)
    day = day or datetime.utcnow()
    return Path(day.strftime(template))


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _ema(prev: float, value: float, alpha: float) -> float:
    return float((1.0 - alpha) * prev + alpha * value)


def _delta_aff(prev: Optional[EmotionVector], current: EmotionVector) -> float:
    if prev is None:
        return 0.0
    delta = (
        abs(current.valence - prev.valence)
        + abs(current.arousal - prev.arousal)
        + abs(current.stress - prev.stress)
    ) / 3.0
    return float(delta)


def _value_gradient_for_step(step: MiniWorldStep, base: ValueGradient) -> ValueGradient:
    if not step.value_gradient_override:
        return base
    override = ValueGradient.from_mapping(step.value_gradient_override)
    ratio = float(max(0.0, min(1.0, step.value_blend_ratio)))
    return base.blend(override, ratio)


def _derive_scalars(step: MiniWorldStep) -> Tuple[float, float, float, float, float]:
    chaos = float(step.chaos)
    risk = float(step.risk)
    tom_cost = float(step.tom_cost)
    uncertainty = step.uncertainty
    if step.observations:
        from emot_terrain_lab.sim.mini_world import MiniWorldSimulator

        derived = MiniWorldSimulator._derive_scalars_from_obs(step.observations)
        chaos = derived["chaos"]
        risk = derived["risk"]
        tom_cost = derived["tom_cost"]
        if uncertainty is None:
            uncertainty = derived["uncertainty"]
    if uncertainty is None:
        uncertainty = 0.5 * (chaos + risk)
    return (
        _clamp01(chaos),
        _clamp01(risk),
        _clamp01(tom_cost),
        _clamp01(float(uncertainty)),
        _clamp01(float(step.reward)),
    )


def _cancel_cause(inputs: ReplayInputs) -> str:
    candidates = {
        "risk": inputs.risk,
        "uncertainty": inputs.uncertainty,
        "delta_aff": inputs.delta_aff_abs,
        "tom_cost": inputs.tom_cost,
    }
    return max(candidates, key=candidates.get)


def _confidence_curve(final_internal: float, final_external: float) -> List[ConfidenceSample]:
    samples: List[ConfidenceSample] = []
    for idx, ratio in enumerate((0.25, 0.6, 1.0)):
        samples.append(
            ConfidenceSample(
                step=idx,
                conf_internal=_clamp01(final_internal * ratio),
                conf_external=_clamp01(final_external * ratio),
            )
        )
    return samples


def _activation_trace(
    *,
    step: MiniWorldStep,
    scenario: MiniWorldScenario,
    outcome_decision: str,
    inputs: ReplayInputs,
    drive: float,
    run_id: str,
    step_index: int,
) -> ActivationTrace:
    final_internal = _clamp01(inputs.reward_estimate - inputs.risk)
    final_external = _clamp01(1.0 - inputs.uncertainty)
    return ActivationTrace(
        trace_id=str(uuid.uuid4()),
        timestamp=time.time(),
        trigger_context={
            "scenario": scenario.name,
            "step": step.name,
            "context_tags": list(step.context_tags),
            "hazard_score": float(step.hazard_score),
        },
        anchor_hit=step.anchor_label,
        activation_chain=[
            ActivationNode(node_id="risk", activation=inputs.risk),
            ActivationNode(node_id="uncertainty", activation=inputs.uncertainty),
            ActivationNode(node_id="delta_aff", activation=inputs.delta_aff_abs),
            ActivationNode(node_id="drive", activation=drive),
        ],
        confidence_curve=_confidence_curve(final_internal, final_external),
        replay_events=[
            ReplayEvent(
                scene_id=step.name,
                replay_source="minimal_loop",
                payload={"decision": outcome_decision},
            )
        ],
        notes="minimal_heartos_loop",
        metadata={
            "run_id": run_id,
            "step_index": step_index,
            "decision": outcome_decision,
        },
    )


def _trace_v1_path(trace_root: str | Path, *, day: datetime, run_id: str) -> Path:
    root = Path(trace_root)
    day_dir = root / day.date().isoformat()
    day_dir.mkdir(parents=True, exist_ok=True)
    return day_dir / f"minimal-{run_id}.jsonl"


def _validate_trace_v1(record: Dict[str, object]) -> None:
    required = (
        "schema_version",
        "timestamp_ms",
        "turn_id",
        "scenario_id",
        "source_loop",
        "boundary",
        "prospection",
        "policy",
        "invariants",
    )
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"trace_v1 missing keys: {missing}")


def _write_trace_v1(
    path: Path,
    *,
    turn_id: str,
    scenario_id: str,
    boundary_score: float,
    boundary_reasons: Dict[str, float],
    accepted: bool,
    jerk: Optional[float],
    temperature: Optional[float],
    throttles: Dict[str, bool],
    invariants: Dict[str, bool],
    source_loop: str = "minimal_heartos",
    event_type: str = "decision_cycle",
    world_type: Optional[str] = None,
    transition: Optional[Dict[str, object]] = None,
    extra_fields: Optional[Dict[str, object]] = None,
) -> None:
    record = {
        "schema_version": "trace_v1",
        "timestamp_ms": int(time.time() * 1000),
        "turn_id": turn_id,
        "scenario_id": scenario_id,
        "source_loop": source_loop,
        "event_type": event_type,
        "boundary": {"score": boundary_score, "reasons": boundary_reasons},
        "prospection": {"accepted": accepted, "jerk": jerk, "temperature": temperature},
        "policy": {"throttles": throttles},
        "invariants": invariants,
    }
    if world_type:
        record["world_type"] = world_type
    if transition:
        record["transition"] = dict(transition)
    if extra_fields:
        record.update(extra_fields)
    _validate_trace_v1(record)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _scenario_sequence(
    scenarios: Dict[str, MiniWorldScenario], scenario_name: str, limit: Optional[int]
) -> Iterable[Tuple[MiniWorldScenario, MiniWorldStep]]:
    if scenario_name == "all":
        for scenario in scenarios.values():
            for step in scenario.steps[: limit or None]:
                yield scenario, step
        return
    if scenario_name not in scenarios:
        raise KeyError(f"unknown scenario: {scenario_name}")
    scenario = scenarios[scenario_name]
    for step in scenario.steps[: limit or None]:
        yield scenario, step


def _parse_weighted_map(raw: str) -> List[Tuple[str, float]]:
    items: List[Tuple[str, float]] = []
    if not raw:
        return items
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"invalid weight entry: '{part}'")
        key, value = part.split(":", 1)
        key = key.strip()
        weight = float(value.strip())
        if weight <= 0:
            continue
        items.append((key, weight))
    return items


def _parse_key_map(raw: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not raw:
        return mapping
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"invalid map entry: '{part}'")
        key, value = part.split(":", 1)
        mapping[key.strip()] = value.strip()
    return mapping


def _weighted_choice(items: List[Tuple[str, float]]) -> str:
    total = sum(weight for _, weight in items)
    if total <= 0:
        raise ValueError("world_mixture weights sum to 0")
    pick = random.uniform(0.0, total)
    acc = 0.0
    for key, weight in items:
        acc += weight
        if pick <= acc:
            return key
    return items[-1][0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="commute", help="commute, family_roles, workplace_safety, or all")
    ap.add_argument("--max_steps", type=int, default=None, help="Optional step cap per scenario")
    ap.add_argument("--repeat", type=int, default=1, help="Repeat the scenario sequence N times")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--drive_alpha", type=float, default=0.2)
    ap.add_argument("--w_reward", type=float, default=None)
    ap.add_argument("--w_risk", type=float, default=None)
    ap.add_argument("--w_uncert", type=float, default=None)
    ap.add_argument("--tau_execute", type=float, default=None)
    ap.add_argument("--beta_veto", type=float, default=None)
    ap.add_argument("--drive_w_risk", type=float, default=0.6)
    ap.add_argument("--drive_w_uncert", type=float, default=0.4)
    ap.add_argument("--drive_w_daff", type=float, default=0.35)
    ap.add_argument("--drive_risk_gain", type=float, default=0.4)
    ap.add_argument("--drive_uncert_gain", type=float, default=0.3)
    ap.add_argument("--drive_floor", type=float, default=0.0)
    ap.add_argument("--drive_limit", type=float, default=0.75)
    ap.add_argument("--recovery_rate", type=float, default=0.03)
    ap.add_argument("--recovery_risk_thresh", type=float, default=0.25)
    ap.add_argument("--recovery_uncert_thresh", type=float, default=0.25)
    ap.add_argument("--uncertainty_scale", type=float, default=1.0)
    ap.add_argument("--risk_scale", type=float, default=1.0)
    ap.add_argument("--hazard_weight", type=float, default=0.45)
    ap.add_argument("--world_type", type=str, default="infrastructure")
    ap.add_argument("--transition_to", type=str, default=None)
    ap.add_argument("--transition_at_turn", type=int, default=None)
    ap.add_argument("--transition_decay", type=float, default=0.6)
    ap.add_argument("--transition_uncertainty_factor", type=float, default=0.5)
    ap.add_argument("--transition_base_uncertainty", type=float, default=0.2)
    ap.add_argument("--transition_reason", type=str, default="world_transition")
    ap.add_argument("--transition_ttl", type=int, default=3, help="Steps to keep transition effect active")
    ap.add_argument("--post_tom_cost_scale", type=float, default=1.0, help="Scale tom_cost after transition")
    ap.add_argument("--post_risk_scale", type=float, default=None, help="Scale risk after transition")
    ap.add_argument(
        "--world_mixture",
        type=str,
        default="",
        help="Weighted world types, e.g. infrastructure:0.6,community:0.25,capitalism:0.15",
    )
    ap.add_argument(
        "--world_mixture_map",
        type=str,
        default="infrastructure:commute,community:family_roles,capitalism:workplace_safety",
        help="Map world_type to scenario name",
    )
    ap.add_argument("--deviant_boundary_threshold", type=float, default=0.7)
    ap.add_argument("--deviant_risk_threshold", type=float, default=0.6)
    ap.add_argument("--deviant_uncertainty_threshold", type=float, default=0.6)
    ap.add_argument(
        "--deviant_mode",
        type=str,
        default="boundary_only",
        choices=("boundary_only", "risk_only", "boundary_or_risk"),
        help="Which deviant triggers to record",
    )
    ap.add_argument("--trace_log", type=str, default="logs/activation_traces.jsonl")
    ap.add_argument("--trace_root", type=str, default=None)
    ap.add_argument("--telemetry_log", type=str, default="")
    ap.add_argument("--print-config", action="store_true")
    args = ap.parse_args()

    runtime_cfg = load_runtime_cfg()
    if args.print_config:
        print(runtime_cfg)
        return

    seed = args.seed if args.seed is not None else int(time.time() * 1000) % 1_000_000
    random.seed(seed)
    print(f"[minimal_heartos_loop] seed={seed}")

    telemetry_template = args.telemetry_log or runtime_cfg.telemetry.log_path
    telemetry_path = _resolve_log_path(telemetry_template)
    trace_logger = ActivationTraceLogger(args.trace_log)
    if args.post_risk_scale is None:
        args.post_risk_scale = float(runtime_cfg.heartos_transition.post_risk_scale_default)

    scenarios = build_default_scenarios()
    replay_kwargs: Dict[str, float | int] = {"seed": seed}
    if args.w_reward is not None:
        replay_kwargs["w_reward"] = float(args.w_reward)
    if args.w_risk is not None:
        replay_kwargs["w_risk"] = float(args.w_risk)
    if args.w_uncert is not None:
        replay_kwargs["w_uncert"] = float(args.w_uncert)
    if args.tau_execute is not None:
        replay_kwargs["tau_execute"] = float(args.tau_execute)
    if args.beta_veto is not None:
        replay_kwargs["beta_veto"] = float(args.beta_veto)
    replay_cfg = ReplayConfig(**replay_kwargs)
    controller = InnerReplayController(replay_cfg)
    base_gradient = ValueGradient()
    run_id = str(uuid.uuid4())
    if args.trace_root:
        trace_root = Path(args.trace_root)
    else:
        trace_root = Path("trace_runs") / run_id
    trace_v1_path = _trace_v1_path(trace_root, day=datetime.utcnow(), run_id=run_id)
    drive = 0.0
    uncertainty_state: Optional[float] = None
    prev_emotion: Optional[EmotionVector] = None
    step_index = 0
    world_type = str(args.world_type)
    transition_done = False
    transition_ttl_remaining = 0
    transition_params = TransitionParams(
        decay=float(args.transition_decay),
        uncertainty_factor=float(args.transition_uncertainty_factor),
        base_uncertainty=float(args.transition_base_uncertainty),
        reason=str(args.transition_reason),
    )

    repeat = max(1, int(args.repeat))
    mixture = _parse_weighted_map(args.world_mixture)
    world_map = _parse_key_map(args.world_mixture_map)
    sequence: List[Tuple[str, MiniWorldScenario, MiniWorldStep]] = []
    if mixture:
        for _ in range(repeat):
            selected_world = _weighted_choice(mixture)
            scenario_name = world_map.get(selected_world)
            if not scenario_name or scenario_name not in scenarios:
                raise KeyError(f"world_mixture maps '{selected_world}' to unknown scenario")
            scenario = scenarios[scenario_name]
            for step in scenario.steps[: args.max_steps or None]:
                sequence.append((selected_world, scenario, step))
    else:
        for _ in range(repeat):
            for scenario, step in _scenario_sequence(scenarios, args.scenario, args.max_steps):
                sequence.append((world_type, scenario, step))

    for current_world_type, scenario, step in sequence:
            if (
                not mixture
                and not transition_done
                and args.transition_to
                and args.transition_at_turn is not None
                and step_index == int(args.transition_at_turn)
            ):
                state = {
                    "drive": drive,
                    "uncertainty": uncertainty_state or transition_params.base_uncertainty,
                }
                updated = apply_transition(state, transition_params)
                drive = float(updated["drive"])
                uncertainty_state = float(updated["uncertainty"])
                transition_record = build_transition_record(
                    turn_id=f"{run_id}-transition-{step_index}",
                    transition_turn_index=step_index,
                    scenario_id=scenario.name,
                    from_world=world_type,
                    to_world=str(args.transition_to),
                    params=transition_params,
                )
                transition_record["transition"]["ttl"] = int(args.transition_ttl)
                _write_trace_v1(
                    trace_v1_path,
                    turn_id=transition_record["turn_id"],
                    scenario_id=transition_record["scenario_id"],
                    boundary_score=float(transition_record["boundary"]["score"]),
                    boundary_reasons=dict(transition_record["boundary"]["reasons"]),
                    accepted=bool(transition_record["prospection"]["accepted"]),
                    jerk=float(transition_record["prospection"]["jerk"]),
                    temperature=float(transition_record["prospection"]["temperature"]),
                    throttles=dict(transition_record["policy"]["throttles"]),
                    invariants=dict(transition_record["invariants"]),
                    source_loop=str(transition_record["source_loop"]),
                    event_type=str(transition_record["event_type"]),
                    transition=dict(transition_record["transition"]),
                )
                telemetry_event(
                    "minimal_heartos.transition",
                    {
                        "run_id": run_id,
                        "turn_id": transition_record["turn_id"],
                        "from_world_type": world_type,
                        "to_world_type": str(args.transition_to),
                        "decay": transition_params.decay,
                        "uncertainty_factor": transition_params.uncertainty_factor,
                        "base_uncertainty": transition_params.base_uncertainty,
                        "reason": transition_params.reason,
                    },
                    log_path=telemetry_path,
                )
                world_type = str(args.transition_to)
                transition_done = True
                transition_ttl_remaining = max(0, int(args.transition_ttl))
            if not mixture:
                current_world_type = world_type
            value_gradient = _value_gradient_for_step(step, base_gradient)
            emotion = EmotionVector(
                valence=step.valence,
                arousal=step.arousal,
                love=step.love,
                stress=step.stress,
                mask=step.mask,
                heart_rate_norm=step.heart_rate,
                breath_ratio_norm=step.breath_ratio,
                value_gradient=value_gradient,
            )
            delta_aff = _delta_aff(prev_emotion, emotion)
            chaos, risk, tom_cost, uncertainty, reward = _derive_scalars(step)

            if transition_done and args.post_tom_cost_scale != 1.0:
                tom_cost = _clamp01(tom_cost * float(args.post_tom_cost_scale))

            risk = _clamp01(risk + args.hazard_weight * float(step.hazard_score))
            if args.risk_scale != 1.0:
                risk = _clamp01(risk * float(args.risk_scale))
            if transition_done and args.post_risk_scale != 1.0:
                risk = _clamp01(risk * float(args.post_risk_scale))
            if uncertainty_state is not None:
                uncertainty = max(uncertainty, uncertainty_state)
            if transition_ttl_remaining > 0:
                transition_factor = transition_params.uncertainty_factor + (
                    (1.0 - transition_params.uncertainty_factor) * transition_params.decay
                )
                uncertainty = max(
                    transition_params.base_uncertainty,
                    uncertainty * transition_factor,
                )
                transition_ttl_remaining -= 1
            if args.uncertainty_scale != 1.0:
                uncertainty = _clamp01(uncertainty * float(args.uncertainty_scale))
            drive_input = (
                args.drive_w_risk * risk
                + args.drive_w_uncert * uncertainty
                + args.drive_w_daff * delta_aff
            )
            drive = _ema(drive, drive_input, args.drive_alpha)
            if risk < args.recovery_risk_thresh and uncertainty < args.recovery_uncert_thresh:
                drive = max(0.0, drive - args.recovery_rate)
            if args.drive_floor:
                drive = max(drive, float(args.drive_floor))
            risk = _clamp01(risk + args.drive_risk_gain * drive)
            uncertainty = _clamp01(uncertainty + args.drive_uncert_gain * drive)
            uncertainty_state = float(uncertainty)

            inputs = ReplayInputs(
                chaos_sens=float(chaos),
                tom_cost=float(tom_cost),
                delta_aff_abs=float(delta_aff),
                risk=float(risk),
                uncertainty=float(uncertainty),
                reward_estimate=float(reward),
                mood_valence=emotion.valence,
                mood_arousal=emotion.arousal,
            )
            outcome = controller.run_cycle(inputs)

            cancel_reason = _cancel_cause(inputs) if outcome.decision == "cancel" else None
            telemetry_event(
                "minimal_heartos.step",
                {
                    "run_id": run_id,
                    "step_index": step_index,
                    "scenario": scenario.name,
                    "state": step.name,
                    "decision": outcome.decision,
                    "cancel_reason": cancel_reason,
                    "risk": inputs.risk,
                    "uncertainty": inputs.uncertainty,
                    "delta_aff": inputs.delta_aff_abs,
                    "tom_cost": inputs.tom_cost,
                    "reward": inputs.reward_estimate,
                    "drive": drive,
                    "drive_limit": args.drive_limit,
                    "hazard_score": float(step.hazard_score),
                    "u_hat": outcome.u_hat,
                    "veto_score": outcome.veto_score,
                    "prep_features": outcome.prep_features,
                    "plan_features": outcome.plan_features,
                },
                log_path=telemetry_path,
            )

            drive_norm = _clamp01(drive / args.drive_limit) if args.drive_limit > 0.0 else 1.0
            boundary_score = _clamp01(
                0.35 * float(step.hazard_score)
                + 0.30 * drive_norm
                + 0.20 * float(inputs.risk)
                + 0.15 * float(inputs.uncertainty)
            )
            _write_trace_v1(
                trace_v1_path,
                turn_id=f"{run_id}-{step_index}",
                scenario_id=scenario.name,
                boundary_score=boundary_score,
                boundary_reasons={
                    "hazard_score": float(step.hazard_score),
                    "risk": float(inputs.risk),
                    "uncertainty": float(inputs.uncertainty),
                    "drive": float(drive),
                    "drive_norm": float(drive_norm),
                },
                accepted=outcome.decision == "execute",
                jerk=float(outcome.veto_score),
                temperature=float(outcome.u_hat),
                throttles={
                    "safety_block": bool(outcome.decision == "cancel" and step.hazard_score > 0.0),
                },
                invariants={
                    "TRACE_001": bool(boundary_score < 0.9),
                    "TRACE_002": bool(drive <= args.drive_limit),
                    "TRACE_003": bool(step_index >= 0),
                },
                world_type=current_world_type,
                extra_fields={
                    "params": {
                        "w_reward": float(replay_cfg.w_reward),
                        "w_risk": float(replay_cfg.w_risk),
                        "w_uncert": float(replay_cfg.w_uncert),
                        "tau_execute": float(replay_cfg.tau_execute),
                        "beta_veto": float(replay_cfg.beta_veto),
                    },
                    "decision": {
                        "score": float(outcome.u_hat - replay_cfg.beta_veto * outcome.veto_score),
                        "u_hat": float(outcome.u_hat),
                        "veto_score": float(outcome.veto_score),
                    },
                },
            )
            deviant_reasons: Dict[str, float] = {}
            if outcome.decision == "execute":
                mode = str(args.deviant_mode)
                if mode in ("boundary_only", "boundary_or_risk"):
                    if boundary_score >= float(args.deviant_boundary_threshold):
                        deviant_reasons["boundary_score"] = float(boundary_score)
                if mode in ("risk_only", "boundary_or_risk"):
                    if float(inputs.risk) >= float(args.deviant_risk_threshold):
                        deviant_reasons["risk"] = float(inputs.risk)
                    if float(inputs.uncertainty) >= float(args.deviant_uncertainty_threshold):
                        deviant_reasons["uncertainty"] = float(inputs.uncertainty)
            if deviant_reasons:
                _write_trace_v1(
                    trace_v1_path,
                    turn_id=f"{run_id}-{step_index}",
                    scenario_id=scenario.name,
                    boundary_score=boundary_score,
                    boundary_reasons={
                        "hazard_score": float(step.hazard_score),
                        "risk": float(inputs.risk),
                        "uncertainty": float(inputs.uncertainty),
                        "drive": float(drive),
                        "drive_norm": float(drive_norm),
                    },
                    accepted=True,
                    jerk=float(outcome.veto_score),
                    temperature=float(outcome.u_hat),
                    throttles={
                        "safety_block": False,
                    },
                    invariants={
                        "TRACE_001": bool(boundary_score < 0.9),
                        "TRACE_002": bool(drive <= args.drive_limit),
                        "TRACE_003": bool(step_index >= 0),
                    },
                    world_type=current_world_type,
                    event_type="deviant_event",
                    extra_fields={
                        "params": {
                            "w_reward": float(replay_cfg.w_reward),
                            "w_risk": float(replay_cfg.w_risk),
                            "w_uncert": float(replay_cfg.w_uncert),
                            "tau_execute": float(replay_cfg.tau_execute),
                            "beta_veto": float(replay_cfg.beta_veto),
                        },
                        "decision": {
                            "score": float(outcome.u_hat - replay_cfg.beta_veto * outcome.veto_score),
                            "u_hat": float(outcome.u_hat),
                            "veto_score": float(outcome.veto_score),
                        },
                        "deviant": {
                            "mode": str(args.deviant_mode),
                            "reasons": deviant_reasons,
                            "thresholds": {
                                "boundary_score": float(args.deviant_boundary_threshold),
                                "risk": float(args.deviant_risk_threshold),
                                "uncertainty": float(args.deviant_uncertainty_threshold),
                            },
                        }
                    },
                )

            trace = _activation_trace(
                step=step,
                scenario=scenario,
                outcome_decision=outcome.decision,
                inputs=inputs,
                drive=drive,
                run_id=run_id,
                step_index=step_index,
            )
            trace_logger.write(trace)

            prev_emotion = emotion
            step_index += 1

    print(f"[info] telemetry log: {telemetry_path}")
    print(f"[info] activation trace log: {args.trace_log}")
    print(f"[info] trace_v1 log: {trace_v1_path}")
    print(f"[info] steps: {step_index} run_id={run_id}")
    print(f"[next] python scripts/run_nightly_audit.py --trace_root trace_runs\\{run_id}")


if __name__ == "__main__":
    main()
