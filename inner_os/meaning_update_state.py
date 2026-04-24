from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


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
class MeaningUpdateState:
    state: str
    self_update: str
    relation_update: str
    relation_frame: str
    relation_key: str
    relation_meta_type: str
    causal_frame: str
    causal_key: str
    world_update: str
    memory_update: str
    recall_anchor: str
    memory_resonance: float
    preserve_guard: str
    confidence: float
    fragility: float
    dominant_inputs: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "self_update": self.self_update,
            "relation_update": self.relation_update,
            "relation_frame": self.relation_frame,
            "relation_key": self.relation_key,
            "relation_meta_type": self.relation_meta_type,
            "causal_frame": self.causal_frame,
            "causal_key": self.causal_key,
            "world_update": self.world_update,
            "memory_update": self.memory_update,
            "recall_anchor": self.recall_anchor,
            "memory_resonance": round(self.memory_resonance, 4),
            "preserve_guard": self.preserve_guard,
            "confidence": round(self.confidence, 4),
            "fragility": round(self.fragility, 4),
            "dominant_inputs": list(self.dominant_inputs),
        }


def derive_meaning_update_state(
    *,
    appraisal_state: Mapping[str, Any],
    recent_dialogue_state: Mapping[str, Any] | None = None,
    discussion_thread_state: Mapping[str, Any] | None = None,
    live_engagement_state: Mapping[str, Any] | None = None,
    memory_dynamics_state: Mapping[str, Any] | None = None,
) -> MeaningUpdateState:
    appraisal = dict(appraisal_state or {})
    recent = dict(recent_dialogue_state or {})
    discussion = dict(discussion_thread_state or {})
    live = dict(live_engagement_state or {})
    memory = dict(memory_dynamics_state or {})

    appraisal_name = _text(appraisal.get("state"))
    moment_event = _text(appraisal.get("moment_event"))
    shared_shift = _text(appraisal.get("shared_shift"))
    easing_shift = _float01(appraisal.get("easing_shift"))
    fragility = _float01(appraisal.get("fragility"))
    memory_mode = _text(appraisal.get("memory_mode")) or _text(memory.get("dominant_mode"))
    recall_anchor = _text(appraisal.get("recall_anchor")) or _text(memory.get("recall_anchor"))
    dominant_relation_type = _text(appraisal.get("dominant_relation_type")) or _text(memory.get("dominant_relation_type"))
    dominant_relation_key = _text(appraisal.get("dominant_relation_key"))
    relation_meta_type = _text(appraisal.get("relation_meta_type"))
    dominant_causal_type = _text(appraisal.get("dominant_causal_type")) or _text(memory.get("dominant_causal_type"))
    dominant_causal_key = _text(appraisal.get("dominant_causal_key"))
    memory_resonance = max(
        _float01(appraisal.get("memory_resonance")),
        _float01(memory.get("activation_confidence")),
    )
    recent_state = _text(recent.get("state"))
    discussion_state = _text(discussion.get("state"))
    live_state = _text(live.get("state"))

    if appraisal_name != "active":
        return MeaningUpdateState(
            state="none",
            self_update="",
            relation_update="",
            relation_frame="",
            relation_key="",
            relation_meta_type="",
            causal_frame="",
            causal_key="",
            world_update="",
            memory_update="",
            recall_anchor="",
            memory_resonance=0.0,
            preserve_guard="",
            confidence=0.0,
            fragility=0.0,
            dominant_inputs=[],
        )

    self_update = "breath_returns"
    relation_update = "shared_room_opens"
    world_update = "moment_not_only_strain"
    if moment_event == "laugh_break":
        self_update = "guard_relaxes_for_moment"
        relation_update = "shared_smile_window"
    elif moment_event == "relief_opening":
        self_update = "breath_returns"
        relation_update = "shared_breathing_room"
    elif moment_event == "pleasant_turn":
        self_update = "attention_turns_open"
        relation_update = "shared_turn_toward_good"
    relation_frame = "ambient_link"
    if dominant_relation_type == "same_anchor":
        relation_frame = "same_anchor_link"
    elif dominant_relation_type == "unfinished_carry":
        relation_frame = "unfinished_link"
    elif dominant_relation_type == "cross_context_bridge":
        relation_frame = "cross_context_bridge"
    elif dominant_relation_type == "recurrent_association":
        relation_frame = "returning_pattern"
    causal_frame = ""
    if dominant_causal_type == "enabled_by":
        causal_frame = "same_anchor_cause"
    elif dominant_causal_type == "reopened_by":
        causal_frame = "unfinished_thread_cause"
    elif dominant_causal_type == "reframed_by":
        causal_frame = "reframing_cause"
    elif dominant_causal_type == "triggered_by":
        causal_frame = "memory_trigger_cause"
    elif dominant_causal_type == "amplified_by":
        causal_frame = "reinforced_cause"
    elif dominant_causal_type == "suppressed_by":
        causal_frame = "guarded_cause"

    if recent_state == "bright_continuity" or discussion_state in {"continuing_thread", "open_thread"}:
        world_update = "topic_not_only_strain"
    elif shared_shift:
        world_update = "momentary_space_opens"
    memory_update = ""
    if recall_anchor and memory_mode == "ignite" and memory_resonance >= 0.34:
        memory_update = "known_thread_returns"
    elif recall_anchor and memory_mode == "reconsolidate" and memory_resonance >= 0.3:
        memory_update = "known_thread_reframes"
    elif recall_anchor and memory_resonance >= 0.28:
        memory_update = "known_thread_stays_near"
    if not memory_update and dominant_relation_type == "unfinished_carry" and recall_anchor:
        memory_update = "unfinished_link_stirs"
    elif not memory_update and dominant_relation_type == "cross_context_bridge" and recall_anchor:
        memory_update = "distant_link_bridges"
    elif not memory_update and dominant_relation_type == "recurrent_association" and recall_anchor:
        memory_update = "pattern_returns"
    if not memory_update and dominant_causal_type == "reopened_by" and recall_anchor:
        memory_update = "unfinished_link_stirs"
    elif not memory_update and dominant_causal_type == "reframed_by" and recall_anchor:
        memory_update = "known_thread_reframes"
    elif not memory_update and dominant_causal_type == "triggered_by" and recall_anchor:
        memory_update = "pattern_returns"
    if memory_update == "known_thread_returns":
        world_update = "small_moment_on_known_thread"
    elif memory_update == "known_thread_reframes":
        world_update = "shared_history_softens"
    elif memory_update == "unfinished_link_stirs":
        world_update = "unfinished_thread_stirs_again"
    elif memory_update == "distant_link_bridges":
        world_update = "distant_echo_connects"
    elif memory_update == "pattern_returns":
        world_update = "familiar_pattern_returns"

    confidence = _float01(
        easing_shift * 0.66
        + memory_resonance * 0.1
        + (0.12 if live_state in {"pickup_comment", "riff_with_comment"} else 0.0)
        + (0.08 if relation_update.startswith("shared_") else 0.0)
        + (0.08 if causal_frame else 0.0)
        - fragility * 0.14
    )
    preserve_guard = "keep_it_small" if fragility >= 0.44 else "do_not_overclaim"
    if memory_update and fragility >= 0.32:
        preserve_guard = "keep_it_small_and_linked"
    if dominant_causal_type == "suppressed_by" and preserve_guard == "do_not_overclaim":
        preserve_guard = "keep_it_small_and_linked"
    dominant_inputs = [
        f"event:{moment_event}",
        f"relation:{relation_update}",
        f"relation_frame:{relation_frame}",
        f"world:{world_update}",
    ]
    if causal_frame:
        dominant_inputs.append(f"causal:{causal_frame}")
    if dominant_causal_key:
        dominant_inputs.append(f"causal_key:{dominant_causal_key}")
    if memory_update:
        dominant_inputs.append(f"memory:{memory_update}")
    if recall_anchor:
        dominant_inputs.append(f"memory_anchor:{recall_anchor}")
    if dominant_relation_type:
        dominant_inputs.append(f"dominant_relation:{dominant_relation_type}")
    if dominant_relation_key:
        dominant_inputs.append(f"relation_key:{dominant_relation_key}")
    if relation_meta_type:
        dominant_inputs.append(f"relation_meta:{relation_meta_type}")
    if live_state:
        dominant_inputs.append(f"live:{live_state}")

    return MeaningUpdateState(
        state="active",
        self_update=self_update,
        relation_update=relation_update,
        relation_frame=relation_frame,
        relation_key=dominant_relation_key,
        relation_meta_type=relation_meta_type,
        causal_frame=causal_frame,
        causal_key=dominant_causal_key,
        world_update=world_update,
        memory_update=memory_update,
        recall_anchor=recall_anchor,
        memory_resonance=memory_resonance,
        preserve_guard=preserve_guard,
        confidence=confidence,
        fragility=fragility,
        dominant_inputs=dominant_inputs,
    )
