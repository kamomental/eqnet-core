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
class UtteranceReasonPacket:
    state: str
    reaction_target: str
    reason_frame: str
    relation_frame: str
    relation_key: str
    causal_frame: str
    causal_key: str
    memory_frame: str
    memory_anchor: str
    offer: str
    preserve: str
    question_policy: str
    tone_hint: str
    dominant_inputs: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "reaction_target": self.reaction_target,
            "reason_frame": self.reason_frame,
            "relation_frame": self.relation_frame,
            "relation_key": self.relation_key,
            "causal_frame": self.causal_frame,
            "causal_key": self.causal_key,
            "memory_frame": self.memory_frame,
            "memory_anchor": self.memory_anchor,
            "offer": self.offer,
            "preserve": self.preserve,
            "question_policy": self.question_policy,
            "tone_hint": self.tone_hint,
            "dominant_inputs": list(self.dominant_inputs),
        }


def derive_utterance_reason_packet(
    *,
    appraisal_state: Mapping[str, Any],
    meaning_update_state: Mapping[str, Any],
    listener_action_state: Mapping[str, Any] | None = None,
    live_engagement_state: Mapping[str, Any] | None = None,
    memory_dynamics_state: Mapping[str, Any] | None = None,
) -> UtteranceReasonPacket:
    appraisal = dict(appraisal_state or {})
    meaning = dict(meaning_update_state or {})
    listener = dict(listener_action_state or {})
    live = dict(live_engagement_state or {})
    memory = dict(memory_dynamics_state or {})

    if _text(appraisal.get("state")) != "active" or _text(meaning.get("state")) != "active":
        return UtteranceReasonPacket(
            state="none",
            reaction_target="",
            reason_frame="",
            relation_frame="",
            relation_key="",
            causal_frame="",
            causal_key="",
            memory_frame="",
            memory_anchor="",
            offer="",
            preserve="",
            question_policy="",
            tone_hint="",
            dominant_inputs=[],
        )

    moment_event = _text(appraisal.get("moment_event"))
    relation_update = _text(meaning.get("relation_update"))
    preserve_guard = _text(meaning.get("preserve_guard"))
    memory_update = _text(meaning.get("memory_update"))
    relation_frame = _text(meaning.get("relation_frame"))
    relation_key = _text(meaning.get("relation_key")) or _text(appraisal.get("dominant_relation_key"))
    causal_frame = _text(meaning.get("causal_frame"))
    causal_key = _text(meaning.get("causal_key")) or _text(appraisal.get("dominant_causal_key"))
    memory_anchor = _text(meaning.get("recall_anchor")) or _text(appraisal.get("recall_anchor")) or _text(memory.get("recall_anchor"))
    memory_mode = _text(appraisal.get("memory_mode")) or _text(memory.get("dominant_mode"))
    memory_resonance = max(
        _float01(meaning.get("memory_resonance")),
        _float01(appraisal.get("memory_resonance")),
        _float01(memory.get("activation_confidence")),
    )
    listener_state = _text(listener.get("state"))
    live_state = _text(live.get("state"))

    reaction_target = "small_shift"
    offer = "brief_ack"
    if moment_event == "laugh_break":
        reaction_target = "small_laugh_moment"
        offer = "brief_shared_smile"
    elif moment_event == "relief_opening":
        reaction_target = "small_relief_moment"
        offer = "brief_relief_ack"
    elif moment_event == "pleasant_turn":
        reaction_target = "small_good_turn"
        offer = "brief_good_turn_ack"

    reason_frame = "acknowledge_small_opening"
    if relation_update.startswith("shared_"):
        reason_frame = "name_shared_shift"
    if relation_frame == "unfinished_link":
        reason_frame = "soften_unfinished_link"
    elif relation_frame == "cross_context_bridge":
        reason_frame = "name_cross_context_echo"
    elif relation_frame == "returning_pattern":
        reason_frame = "name_returning_pattern"
    if causal_frame == "unfinished_thread_cause":
        reason_frame = "soften_unfinished_link"
    elif causal_frame == "reframing_cause":
        reason_frame = "name_cross_context_echo"
    elif causal_frame == "memory_trigger_cause":
        reason_frame = "name_returning_pattern"
    memory_frame = ""
    if memory_update == "known_thread_returns" and memory_anchor:
        memory_frame = "echo_known_thread"
    elif memory_update == "known_thread_reframes" and memory_anchor:
        memory_frame = "name_small_return"
    elif memory_update == "known_thread_stays_near" and memory_anchor:
        memory_frame = "keep_known_thread_near"
    elif memory_update == "unfinished_link_stirs" and memory_anchor:
        memory_frame = "keep_unfinished_link_near"
    elif memory_update == "distant_link_bridges" and memory_anchor:
        memory_frame = "name_distant_link"
    elif memory_update == "pattern_returns" and memory_anchor:
        memory_frame = "echo_returning_pattern"
    elif memory_resonance >= 0.34 and memory_mode in {"ignite", "reconsolidate"}:
        memory_frame = "keep_history_near"

    tone_hint = "soft_ack"
    if listener_state == "warm_laugh_ack" or live_state == "riff_with_comment":
        tone_hint = "chatty_ack"
    elif listener_state == "playful_ack":
        tone_hint = "playful_ack"

    dominant_inputs = [
        f"target:{reaction_target}",
        f"offer:{offer}",
        f"frame:{reason_frame}",
    ]
    if relation_frame:
        dominant_inputs.append(f"relation:{relation_frame}")
    if relation_key:
        dominant_inputs.append(f"relation_key:{relation_key}")
    if causal_frame:
        dominant_inputs.append(f"causal:{causal_frame}")
    if causal_key:
        dominant_inputs.append(f"causal_key:{causal_key}")
    if memory_frame:
        dominant_inputs.append(f"memory:{memory_frame}")
    if memory_anchor:
        dominant_inputs.append(f"memory_anchor:{memory_anchor}")
    if preserve_guard:
        dominant_inputs.append(f"preserve:{preserve_guard}")
    if tone_hint:
        dominant_inputs.append(f"tone:{tone_hint}")

    return UtteranceReasonPacket(
        state="active",
        reaction_target=reaction_target,
        reason_frame=reason_frame,
        relation_frame=relation_frame,
        relation_key=relation_key,
        causal_frame=causal_frame,
        causal_key=causal_key,
        memory_frame=memory_frame,
        memory_anchor=memory_anchor,
        offer=offer,
        preserve=preserve_guard or "do_not_overclaim",
        question_policy="none",
        tone_hint=tone_hint,
        dominant_inputs=dominant_inputs,
    )
