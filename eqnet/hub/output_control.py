from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from eqnet.hub.repair_fsm import RepairEvent, RepairSnapshot
from eqnet.runtime.policy import PolicyPrior


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
    control_applied_at: str = "response_gate_v1"

    def to_fingerprint_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        return {
            "response_style_mode": str(payload["response_style_mode"]),
            "recall_budget_override": int(payload["recall_budget_override"]),
            "safety_strictness": float(payload["safety_strictness"]),
            "temperature_cap": float(payload["temperature_cap"]),
            "repair_state": str(payload["repair_state"]),
            "control_applied_at": str(payload["control_applied_at"]),
        }


def apply_policy_prior(
    policy_prior: PolicyPrior,
    *,
    day_key: str,
    episode_id: str,
    repair_snapshot: RepairSnapshot | None = None,
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
    )
    return apply_repair_overlay(output, repair_snapshot=repair_snapshot)


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
            control_applied_at=output.control_applied_at,
        )
    return OutputControl(
        response_style_mode="repair",
        recall_budget_override=min(output.recall_budget_override, 1),
        safety_strictness=_clamp01(max(output.safety_strictness, 0.75)),
        temperature_cap=_clamp01(min(output.temperature_cap, 0.45)),
        repair_state=state.value,
        control_applied_at=output.control_applied_at,
    )
