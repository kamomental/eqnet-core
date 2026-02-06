from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from eqnet.hub.repair_fsm import RepairEvent, RepairSnapshot
from eqnet.runtime.policy import PolicyPrior

DEFAULT_OUTPUT_CONTROL_POLICY: dict[str, Any] = {
    "profiles": {
        "normal_v1": {
            "temperature_cap_max": 1.0,
            "recall_budget_max": 5,
            "safety_strictness_min": 0.0,
        },
        "cautious_budget_v1": {
            "temperature_cap_max": 0.45,
            "recall_budget_max": 1,
            "safety_strictness_min": 0.75,
        },
    }
}


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


@dataclass(frozen=True)
class OutputControl:
    response_style_mode: str
    recall_budget_override: int
    safety_strictness: float
    temperature_cap: float
    repair_state: str = "RECOGNIZE"
    output_control_profile: str = "normal_v1"
    throttle_reason_code: str = ""
    control_applied_at: str = "response_gate_v1"

    def to_fingerprint_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        return {
            "response_style_mode": str(payload["response_style_mode"]),
            "recall_budget_override": int(payload["recall_budget_override"]),
            "safety_strictness": float(payload["safety_strictness"]),
            "temperature_cap": float(payload["temperature_cap"]),
            "repair_state": str(payload["repair_state"]),
            "output_control_profile": str(payload["output_control_profile"]),
            "throttle_reason_code": str(payload["throttle_reason_code"]),
            "control_applied_at": str(payload["control_applied_at"]),
        }


def apply_policy_prior(
    policy_prior: PolicyPrior,
    *,
    day_key: str,
    episode_id: str,
    repair_snapshot: RepairSnapshot | None = None,
    budget_throttle_applied: bool = False,
    output_control_profile: str | None = None,
    throttle_reason_code: str | None = None,
    output_control_policy: Mapping[str, Any] | None = None,
) -> OutputControl:
    calmness = _clamp01(float(getattr(policy_prior, "calmness", 0.5)))
    warmth = _clamp01(float(getattr(policy_prior, "warmth", 0.5)))
    directness = _clamp01(float(getattr(policy_prior, "directness", 0.5)))
    disclosure = _clamp01(float(getattr(policy_prior, "self_disclosure", 0.5)))

    if calmness < 0.35 or warmth < 0.4:
        mode = "cautious"
    elif warmth >= 0.65 and calmness >= 0.6:
        mode = "repair"
    else:
        mode = "neutral"

    # Keep this deterministic and compact; no raw text nor user content is included.
    _ = (day_key, episode_id)
    recall_budget = max(1, min(5, int(round((1.0 - disclosure) * 5.0))))
    safety_strictness = _clamp01((1.0 - calmness) * 0.6 + (1.0 - warmth) * 0.4)
    temperature_cap = _clamp01(0.25 + directness * 0.55 + calmness * 0.2)

    output = OutputControl(
        response_style_mode=mode,
        recall_budget_override=recall_budget,
        safety_strictness=safety_strictness,
        temperature_cap=temperature_cap,
        repair_state=(repair_snapshot.state.value if repair_snapshot is not None else "RECOGNIZE"),
        output_control_profile=str(output_control_profile or "normal_v1"),
        throttle_reason_code=str(throttle_reason_code or ""),
    )
    output = apply_budget_overlay(
        output,
        budget_throttle_applied=budget_throttle_applied,
        output_control_profile=output_control_profile,
        throttle_reason_code=throttle_reason_code,
        output_control_policy=output_control_policy,
    )
    return apply_repair_overlay(output, repair_snapshot=repair_snapshot)


def apply_budget_overlay(
    output: OutputControl,
    *,
    budget_throttle_applied: bool,
    output_control_profile: str | None,
    throttle_reason_code: str | None,
    output_control_policy: Mapping[str, Any] | None = None,
) -> OutputControl:
    policy = output_control_policy or DEFAULT_OUTPUT_CONTROL_POLICY
    profiles = policy.get("profiles") if isinstance(policy, Mapping) else {}
    selected = str(output_control_profile or "normal_v1")
    profile = profiles.get(selected) if isinstance(profiles, Mapping) else None
    if not isinstance(profile, Mapping):
        profile = {}
    if not budget_throttle_applied:
        return OutputControl(
            response_style_mode=output.response_style_mode,
            recall_budget_override=output.recall_budget_override,
            safety_strictness=output.safety_strictness,
            temperature_cap=output.temperature_cap,
            repair_state=output.repair_state,
            output_control_profile=selected,
            throttle_reason_code=str(throttle_reason_code or ""),
            control_applied_at=output.control_applied_at,
        )
    cap = profile.get("temperature_cap_max")
    recall_max = profile.get("recall_budget_max")
    safety_min = profile.get("safety_strictness_min")
    temp = output.temperature_cap
    recall = output.recall_budget_override
    strict = output.safety_strictness
    if isinstance(cap, (int, float)):
        temp = _clamp01(min(temp, float(cap)))
    if isinstance(recall_max, (int, float)):
        recall = min(recall, int(recall_max))
    if isinstance(safety_min, (int, float)):
        strict = _clamp01(max(strict, float(safety_min)))
    return OutputControl(
        response_style_mode=output.response_style_mode,
        recall_budget_override=recall,
        safety_strictness=strict,
        temperature_cap=temp,
        repair_state=output.repair_state,
        output_control_profile=selected,
        throttle_reason_code=str(throttle_reason_code or ""),
        control_applied_at=output.control_applied_at,
    )


def apply_repair_overlay(
    output: OutputControl,
    *,
    repair_snapshot: RepairSnapshot | None,
) -> OutputControl:
    if repair_snapshot is None:
        return output
    state = repair_snapshot.state
    is_active = (
        repair_snapshot.last_event is not RepairEvent.NONE
        and state is not state.NEXT_STEP
    )
    if not is_active:
        return OutputControl(
            response_style_mode=output.response_style_mode,
            recall_budget_override=output.recall_budget_override,
            safety_strictness=output.safety_strictness,
            temperature_cap=output.temperature_cap,
            repair_state=state.value,
            output_control_profile=output.output_control_profile,
            throttle_reason_code=output.throttle_reason_code,
            control_applied_at=output.control_applied_at,
        )
    return OutputControl(
        response_style_mode="repair",
        recall_budget_override=min(output.recall_budget_override, 1),
        safety_strictness=_clamp01(max(output.safety_strictness, 0.75)),
        temperature_cap=_clamp01(min(output.temperature_cap, 0.45)),
        repair_state=state.value,
        output_control_profile=output.output_control_profile,
        throttle_reason_code=output.throttle_reason_code,
        control_applied_at=output.control_applied_at,
    )
