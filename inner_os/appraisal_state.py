from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


def _float01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(0.0, min(1.0, numeric))


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


@dataclass(frozen=True)
class AppraisalState:
    state: str
    background_state: str
    moment_event: str
    shared_shift: str
    dominant_relation_type: str
    dominant_relation_key: str
    relation_meta_type: str
    dominant_causal_type: str
    dominant_causal_key: str
    memory_mode: str
    recall_anchor: str
    memory_resonance: float
    easing_shift: float
    continuity_tension: float
    fragility: float
    dominant_inputs: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "background_state": self.background_state,
            "moment_event": self.moment_event,
            "shared_shift": self.shared_shift,
            "dominant_relation_type": self.dominant_relation_type,
            "dominant_relation_key": self.dominant_relation_key,
            "relation_meta_type": self.relation_meta_type,
            "dominant_causal_type": self.dominant_causal_type,
            "dominant_causal_key": self.dominant_causal_key,
            "memory_mode": self.memory_mode,
            "recall_anchor": self.recall_anchor,
            "memory_resonance": round(self.memory_resonance, 4),
            "easing_shift": round(self.easing_shift, 4),
            "continuity_tension": round(self.continuity_tension, 4),
            "fragility": round(self.fragility, 4),
            "dominant_inputs": list(self.dominant_inputs),
        }


def derive_appraisal_state(
    *,
    current_focus: str,
    current_risks: Sequence[str],
    self_state: Mapping[str, Any],
    recent_dialogue_state: Mapping[str, Any] | None = None,
    issue_state: Mapping[str, Any] | None = None,
    shared_moment_state: Mapping[str, Any] | None = None,
    lightness_budget_state: Mapping[str, Any] | None = None,
    memory_dynamics_state: Mapping[str, Any] | None = None,
) -> AppraisalState:
    recent = dict(recent_dialogue_state or {})
    issue = dict(issue_state or {})
    shared_moment = dict(shared_moment_state or {})
    lightness = dict(lightness_budget_state or {})
    current = dict(self_state or {})
    memory = dict(memory_dynamics_state or {})

    recent_state = _text(recent.get("state"))
    issue_state_name = _text(issue.get("state"))
    moment_state = _text(shared_moment.get("state"))
    moment_kind = _text(shared_moment.get("moment_kind"))
    moment_score = _float01(shared_moment.get("score"))
    moment_jointness = _float01(shared_moment.get("jointness"))
    moment_afterglow = _float01(shared_moment.get("afterglow"))
    moment_fragility = _float01(shared_moment.get("fragility"))
    banter_room = _float01(lightness.get("banter_room"))
    memory_mode = _text(memory.get("dominant_mode"))
    recall_anchor = _text(memory.get("recall_anchor"))
    dominant_relation_type = _text(memory.get("dominant_relation_type"))
    dominant_relation_key = ""
    raw_relation_edges = memory.get("relation_edges")
    if isinstance(raw_relation_edges, list) and raw_relation_edges and isinstance(raw_relation_edges[0], Mapping):
        dominant_relation_key = _text(raw_relation_edges[0].get("relation_key"))
    relation_meta_type = ""
    raw_meta_relations = memory.get("meta_relations")
    if isinstance(raw_meta_relations, list) and raw_meta_relations and isinstance(raw_meta_relations[0], Mapping):
        relation_meta_type = _text(raw_meta_relations[0].get("meta_type"))
    dominant_causal_type = _text(memory.get("dominant_causal_type"))
    dominant_causal_key = ""
    raw_causal_edges = memory.get("causal_edges")
    if isinstance(raw_causal_edges, list) and raw_causal_edges and isinstance(raw_causal_edges[0], Mapping):
        dominant_causal_key = _text(raw_causal_edges[0].get("causal_key"))
    palace_topology = _float01(memory.get("palace_topology"))
    monument_salience = _float01(memory.get("monument_salience"))
    ignition_readiness = _float01(memory.get("ignition_readiness"))
    activation_confidence = _float01(memory.get("activation_confidence"))
    memory_tension = _float01(memory.get("memory_tension"))
    recent_strain = _float01(
        current.get("recent_strain")
        or current.get("social_tension")
        or current.get("stress")
        or 0.0
    )
    danger_pressure = any(_text(item) == "danger" for item in list(current_risks or []))
    comment_focus = _text(current_focus).startswith("comment:")
    continuing_thread = recent_state in {
        "continuing_thread",
        "bright_continuity",
        "reopening_thread",
    }

    background_state = "open_background"
    if issue_state_name in {"light_tension", "bright_issue"}:
        background_state = "awkwardness_present"
    elif recent_strain >= 0.32 or recent_state in {"continuing_thread", "reopening_thread"}:
        background_state = "strain_present"
    elif danger_pressure:
        background_state = "guarded_present"
    if (
        background_state == "open_background"
        and recall_anchor
        and memory_mode in {"ignite", "reconsolidate"}
        and (ignition_readiness >= 0.36 or monument_salience >= 0.34)
    ):
        background_state = "remembered_thread_present"
    if background_state == "open_background" and dominant_relation_type == "unfinished_carry":
        background_state = "strain_present"
    if background_state == "open_background" and dominant_causal_type == "reopened_by":
        background_state = "strain_present"

    moment_event = ""
    shared_shift = ""
    if moment_kind == "laugh":
        moment_event = "laugh_break"
        shared_shift = "shared_smile_window"
    elif moment_kind == "relief":
        moment_event = "relief_opening"
        shared_shift = "breathing_room_opened"
    elif moment_kind == "pleasant_surprise":
        moment_event = "pleasant_turn"
        shared_shift = "attention_turns_open"

    memory_resonance = _float01(
        monument_salience * 0.28
        + ignition_readiness * 0.24
        + activation_confidence * 0.18
        + palace_topology * 0.12
        + (0.12 if recall_anchor else 0.0)
        + (0.08 if memory_mode in {"ignite", "reconsolidate"} else 0.0)
        + (0.08 if dominant_relation_type in {"same_anchor", "recurrent_association"} else 0.0)
        + (0.06 if relation_meta_type == "reinforces" else 0.0)
        + (0.06 if dominant_causal_type in {"enabled_by", "triggered_by", "reopened_by"} else 0.0)
        - memory_tension * 0.12
    )
    continuity_tension = _float01(
        recent_strain * 0.58
        + (0.16 if background_state in {"awkwardness_present", "strain_present"} else 0.0)
        + (0.12 if background_state == "remembered_thread_present" else 0.0)
        + (0.08 if continuing_thread else 0.0)
        + (0.12 if danger_pressure else 0.0)
        + memory_tension * 0.08
        + (0.08 if dominant_relation_type == "unfinished_carry" else 0.0)
        + (0.04 if relation_meta_type == "competes_with" else 0.0)
        + (0.08 if dominant_causal_type == "reopened_by" else 0.0)
        + (0.06 if dominant_causal_type == "suppressed_by" else 0.0)
    )
    easing_shift = _float01(
        moment_score * 0.52
        + moment_jointness * 0.18
        + moment_afterglow * 0.16
        + banter_room * 0.14
        + memory_resonance * 0.08
        + (0.06 if comment_focus else 0.0)
        + (0.05 if dominant_relation_type == "cross_context_bridge" else 0.0)
        + (0.06 if dominant_causal_type in {"reframed_by", "amplified_by"} else 0.0)
        - (0.14 if danger_pressure else 0.0)
    )
    fragility = _float01(
        0.18
        + moment_fragility * 0.48
        + continuity_tension * 0.18
        + (0.1 if danger_pressure else 0.0)
    )

    active = (
        moment_state == "shared_moment"
        and bool(moment_event)
        and easing_shift >= 0.34
    )
    dominant_inputs: list[str] = []
    if active:
        dominant_inputs.append(f"moment:{moment_event}")
        dominant_inputs.append(f"shift:{shared_shift}")
    if background_state != "open_background":
        dominant_inputs.append(f"background:{background_state}")
    if continuing_thread:
        dominant_inputs.append("thread:continuing")
    if comment_focus:
        dominant_inputs.append("focus:comment")
    if banter_room >= 0.24:
        dominant_inputs.append("room:banter")
    if memory_resonance >= 0.32:
        dominant_inputs.append(f"memory:{memory_mode or 'ambient'}")
    if recall_anchor:
        dominant_inputs.append(f"memory_anchor:{recall_anchor}")
    if dominant_relation_type:
        dominant_inputs.append(f"relation:{dominant_relation_type}")
    if dominant_relation_key:
        dominant_inputs.append(f"relation_key:{dominant_relation_key}")
    if relation_meta_type:
        dominant_inputs.append(f"relation_meta:{relation_meta_type}")
    if dominant_causal_type:
        dominant_inputs.append(f"causal:{dominant_causal_type}")
    if dominant_causal_key:
        dominant_inputs.append(f"causal_key:{dominant_causal_key}")
    if danger_pressure:
        dominant_inputs.append("risk:danger")

    return AppraisalState(
        state="active" if active else "none",
        background_state=background_state if active else "",
        moment_event=moment_event if active else "",
        shared_shift=shared_shift if active else "",
        dominant_relation_type=dominant_relation_type if active else "",
        dominant_relation_key=dominant_relation_key if active else "",
        relation_meta_type=relation_meta_type if active else "",
        dominant_causal_type=dominant_causal_type if active else "",
        dominant_causal_key=dominant_causal_key if active else "",
        memory_mode=memory_mode if active else "",
        recall_anchor=recall_anchor if active else "",
        memory_resonance=memory_resonance if active else 0.0,
        easing_shift=easing_shift if active else 0.0,
        continuity_tension=continuity_tension if active else 0.0,
        fragility=fragility if active else 0.0,
        dominant_inputs=dominant_inputs if active else [],
    )
