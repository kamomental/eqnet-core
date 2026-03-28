from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SocialExperimentLoopState:
    state: str
    score: float
    scores: dict[str, float]
    winner_margin: float
    hypothesis: str
    expected_signal: str
    stop_rule: str
    probe_intensity: float
    dominant_inputs: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "score": round(self.score, 4),
            "scores": {str(key): round(float(value), 4) for key, value in self.scores.items()},
            "winner_margin": round(self.winner_margin, 4),
            "hypothesis": self.hypothesis,
            "expected_signal": self.expected_signal,
            "stop_rule": self.stop_rule,
            "probe_intensity": round(self.probe_intensity, 4),
            "dominant_inputs": list(self.dominant_inputs),
        }


def derive_social_experiment_loop_state(
    *,
    learning_mode_state: Mapping[str, Any],
    commitment_state: Mapping[str, Any],
    agenda_state: Mapping[str, Any],
    agenda_window_state: Mapping[str, Any],
    body_recovery_guard: Mapping[str, Any],
    protection_mode: Mapping[str, Any],
    grice_guard_state: Mapping[str, Any],
    relational_continuity_state: Mapping[str, Any],
    social_topology_state: Mapping[str, Any],
    self_state: Mapping[str, Any],
    identity_arc_kind: str = "",
) -> SocialExperimentLoopState:
    learning_mode = dict(learning_mode_state or {})
    commitment = dict(commitment_state or {})
    agenda = dict(agenda_state or {})
    agenda_window = dict(agenda_window_state or {})
    body_guard = dict(body_recovery_guard or {})
    protection = dict(protection_mode or {})
    grice = dict(grice_guard_state or {})
    continuity = dict(relational_continuity_state or {})
    topology = dict(social_topology_state or {})
    state_payload = dict(self_state or {})

    learning_state = str(learning_mode.get("state") or "observe_only").strip() or "observe_only"
    learning_score = _float01(learning_mode.get("score"))
    probe_room = _float01(learning_mode.get("probe_room"))
    update_bias = _float01(learning_mode.get("update_bias"))
    commitment_mode = str(commitment.get("state") or "waver").strip() or "waver"
    commitment_target = str(commitment.get("target") or "hold").strip() or "hold"
    commitment_score = _float01(commitment.get("score"))
    agenda_name = str(agenda.get("state") or "hold").strip() or "hold"
    agenda_score = _float01(agenda.get("score"))
    agenda_window_name = str(agenda_window.get("state") or "long_hold").strip() or "long_hold"
    body_guard_state = str(body_guard.get("state") or "open").strip() or "open"
    body_guard_score = _float01(body_guard.get("score"))
    protection_mode_name = str(protection.get("mode") or "").strip()
    protection_strength = _float01(protection.get("strength"))
    grice_state = str(grice.get("state") or "advise_openly").strip() or "advise_openly"
    continuity_state = str(continuity.get("state") or "distant").strip() or "distant"
    continuity_score = _float01(continuity.get("score"))
    topology_state = str(topology.get("state") or "ambient").strip() or "ambient"
    topology_visibility = _float01(topology.get("visibility_pressure"))
    topology_hierarchy = _float01(topology.get("hierarchy_pressure"))
    degraded = bool(state_payload.get("degraded", False))
    carry_focus = str(state_payload.get("social_experiment_focus") or "").strip()
    carry_bias = _float01(state_payload.get("social_experiment_carry_bias"))

    probe_intensity = _clamp01(
        0.08
        + probe_room * 0.38
        + update_bias * 0.18
        + (0.1 if commitment_mode == "commit" else 0.0) * max(commitment_score, 0.5)
        + (0.08 if agenda_name in {"step_forward", "repair"} else 0.0) * max(agenda_score, 0.5)
        - body_guard_score * 0.18
        - protection_strength * (0.16 if protection_mode_name in {"contain", "stabilize", "shield"} else 0.05)
        - topology_visibility * 0.12
        - topology_hierarchy * 0.1
        - (0.12 if degraded else 0.0)
    )

    scores = {
        "watch_and_read": _clamp01(
            0.12
            + learning_score * (0.22 if learning_state == "observe_only" else 0.06)
            + (0.14 if grice_state in {"hold_obvious_advice", "attune_without_repeating"} else 0.0)
            + topology_visibility * 0.08
            + topology_hierarchy * 0.08
            + (0.08 if agenda_window_name in {"next_same_group_window", "next_same_culture_window"} else 0.0)
            - probe_intensity * 0.18
        ),
        "test_small_step": _clamp01(
            0.06
            + probe_intensity * 0.34
            + learning_score * (0.18 if learning_state == "test_small" else 0.06)
            + (0.14 if commitment_target == "step_forward" and commitment_mode == "commit" else 0.0)
            + (0.12 if agenda_name == "step_forward" else 0.06 if agenda_name == "revisit" else 0.0)
            + (0.08 if identity_arc_kind == "growing_edge" else 0.0)
        ),
        "repair_signal_probe": _clamp01(
            0.06
            + probe_intensity * 0.22
            + learning_score * (0.18 if learning_state == "repair_probe" else 0.06)
            + (0.14 if protection_mode_name == "repair" else 0.0)
            + (0.14 if commitment_target in {"repair", "bond_protect"} and commitment_mode == "commit" else 0.0)
            + (0.12 if continuity_state in {"reopening", "holding_thread"} else 0.0)
            + (0.08 if identity_arc_kind in {"repairing_bond", "holding_thread"} else 0.0)
        ),
        "confirm_shared_direction": _clamp01(
            0.05
            + probe_intensity * 0.12
            + learning_score * (0.22 if learning_state == "integrate_and_commit" else 0.04)
            + (0.18 if commitment_mode == "commit" and commitment_target in {"step_forward", "repair"} else 0.0)
            + (0.12 if continuity_state == "co_regulating" else 0.0) * max(continuity_score, 0.5)
            + update_bias * 0.14
        ),
        "hold_probe": _clamp01(
            0.08
            + learning_score * (0.24 if learning_state == "hold_and_wait" else 0.06)
            + body_guard_score * 0.18
            + protection_strength * (0.16 if protection_mode_name in {"contain", "stabilize", "shield"} else 0.08)
            + (0.12 if agenda_window_name in {"next_private_window", "long_hold"} else 0.0)
            + (0.1 if degraded else 0.0)
        ),
    }

    if body_guard_state == "recovery_first" or agenda_window_name == "long_hold":
        scores["hold_probe"] = _clamp01(scores["hold_probe"] + 0.18)
        scores["test_small_step"] *= 0.3
        scores["confirm_shared_direction"] *= 0.48
    if agenda_window_name == "next_private_window":
        scores["hold_probe"] = _clamp01(scores["hold_probe"] + 0.08)
        scores["repair_signal_probe"] = _clamp01(scores["repair_signal_probe"] + 0.04)
        scores["test_small_step"] *= 0.78
    elif agenda_window_name in {"next_same_group_window", "next_same_culture_window"}:
        scores["watch_and_read"] = _clamp01(scores["watch_and_read"] + 0.08)
        scores["test_small_step"] *= 0.84

    if carry_focus in scores and carry_bias > 0.0:
        carry_scale = 0.14
        if carry_focus in {"watch_and_read", "hold_probe"}:
            carry_scale = 0.18
        elif carry_focus == "repair_signal_probe":
            carry_scale = 0.16
        elif carry_focus == "confirm_shared_direction":
            carry_scale = 0.1
        if carry_focus in {"test_small_step", "confirm_shared_direction"} and agenda_window_name in {
            "next_private_window",
            "next_same_group_window",
            "next_same_culture_window",
            "long_hold",
        }:
            carry_scale *= 0.72
        if carry_focus == "confirm_shared_direction" and body_guard_state == "recovery_first":
            carry_scale *= 0.64
        scores[carry_focus] = _clamp01(scores[carry_focus] + carry_bias * carry_scale)

    state, winner_margin = _winner_and_margin(scores)
    hypothesis, expected_signal, stop_rule = _loop_descriptors(
        state=state,
        agenda_window_name=agenda_window_name,
    )
    dominant_inputs = _compact(
        [
            "learning_mode_state",
            "commitment_state" if commitment_mode != "waver" else "",
            "agenda_state" if agenda_name != "hold" else "",
            "agenda_window_state" if agenda_window_name != "now" else "",
            "body_recovery_guard" if body_guard_state != "open" else "",
            "protection_mode" if protection_mode_name else "",
            "grice_guard_state" if grice_state != "advise_openly" else "",
            "relational_continuity_state" if continuity_state != "distant" else "",
            "social_topology_state" if topology_state != "ambient" else "",
            "identity_arc" if identity_arc_kind else "",
            "overnight_social_experiment_carry" if carry_focus and carry_bias > 0.0 else "",
        ]
    )

    return SocialExperimentLoopState(
        state=state,
        score=_clamp01(scores.get(state, 0.0)),
        scores=scores,
        winner_margin=winner_margin,
        hypothesis=hypothesis,
        expected_signal=expected_signal,
        stop_rule=stop_rule,
        probe_intensity=probe_intensity,
        dominant_inputs=dominant_inputs,
    )


def _loop_descriptors(
    *,
    state: str,
    agenda_window_name: str,
) -> tuple[str, str, str]:
    if state == "test_small_step":
        return (
            "small_forward_step_can_hold",
            "partner_keeps_shared_thread_alive",
            "stop_if_load_or_resistance_rises",
        )
    if state == "repair_signal_probe":
        return (
            "small_repair_softens_contact",
            "less_pressure_more_readiness",
            "stop_if_repair_is_not_received",
        )
    if state == "confirm_shared_direction":
        return (
            "shared_direction_is_ready_to_hold",
            "steady_alignment_across_turns",
            "stop_if_alignment_thins_or_context_shifts",
        )
    if state == "hold_probe":
        if agenda_window_name == "next_private_window":
            return (
                "wait_for_private_reentry",
                "safer_private_window",
                "until_private_window",
            )
        return (
            "wait_before_reentry",
            "safer_reentry_window",
            "until_thread_reopens_naturally",
        )
    return (
        "read_if_thread_opens_without_push",
        "more_detail_without_pressure",
        "stop_if_visibility_or_guard_rises",
    )


def _winner_and_margin(scores: Mapping[str, float]) -> tuple[str, float]:
    items = sorted(
        ((str(key), _clamp01(float(value or 0.0))) for key, value in scores.items()),
        key=lambda item: (-item[1], item[0]),
    )
    if not items:
        return "watch_and_read", 0.0
    winner_name, winner_score = items[0]
    runner_up = items[1][1] if len(items) > 1 else 0.0
    return winner_name, round(_clamp01(winner_score - runner_up), 4)


def _compact(values: list[str]) -> list[str]:
    return [value for value in values if value]


def _float01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return _clamp01(numeric)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
