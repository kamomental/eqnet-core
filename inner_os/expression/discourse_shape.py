from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


def _text(value: Any) -> str:
    return str(value or "").strip()


def _float01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(0.0, min(1.0, numeric))


@dataclass(frozen=True)
class DiscourseShape:
    """行為と表面文のあいだに置く談話骨格。"""

    shape_id: str = "reflect_step"
    primary_move: str = "reflect"
    secondary_move: str = "gentle_close"
    sentence_budget: int = 2
    question_budget: int = 0
    anchor_mode: str = "none"
    closing_mode: str = "soft_close"
    energy: str = "neutral"
    brightness: float = 0.0
    playfulness: float = 0.0
    tempo: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape_id": self.shape_id,
            "primary_move": self.primary_move,
            "secondary_move": self.secondary_move,
            "sentence_budget": int(self.sentence_budget),
            "question_budget": int(self.question_budget),
            "anchor_mode": self.anchor_mode,
            "closing_mode": self.closing_mode,
            "energy": self.energy,
            "brightness": round(self.brightness, 4),
            "playfulness": round(self.playfulness, 4),
            "tempo": round(self.tempo, 4),
        }


def coerce_discourse_shape(payload: Mapping[str, Any] | DiscourseShape | None) -> DiscourseShape:
    if isinstance(payload, DiscourseShape):
        return payload
    data = dict(payload or {})
    return DiscourseShape(
        shape_id=_text(data.get("shape_id")) or "reflect_step",
        primary_move=_text(data.get("primary_move")) or "reflect",
        secondary_move=_text(data.get("secondary_move")) or "gentle_close",
        sentence_budget=max(1, int(data.get("sentence_budget") or 2)),
        question_budget=max(0, int(data.get("question_budget") or 0)),
        anchor_mode=_text(data.get("anchor_mode")) or "none",
        closing_mode=_text(data.get("closing_mode")) or "soft_close",
        energy=_text(data.get("energy")) or "neutral",
        brightness=_float01(data.get("brightness")),
        playfulness=_float01(data.get("playfulness")),
        tempo=_float01(data.get("tempo")),
    )


def derive_discourse_shape(
    *,
    content_sequence: Sequence[Mapping[str, Any]] | None = None,
    turn_delta: Mapping[str, Any] | None = None,
    surface_context_packet: Mapping[str, Any] | None = None,
) -> DiscourseShape:
    acts = [
        _text(item.get("act"))
        for item in (content_sequence or [])
        if isinstance(item, Mapping) and _text(item.get("act"))
    ]
    delta = dict(turn_delta or {})
    packet = dict(surface_context_packet or {})
    response_role = dict(packet.get("response_role") or {})
    constraints = dict(packet.get("constraints") or {})
    shared_core = dict(packet.get("shared_core") or {})
    profile = dict(packet.get("surface_profile") or {})
    source_state = dict(packet.get("source_state") or {})

    kind = _text(delta.get("kind"))
    conversation_phase = _text(packet.get("conversation_phase"))
    preferred_act = _text(delta.get("preferred_act")) or _text(response_role.get("primary"))
    anchor = _text(delta.get("anchor_hint")) or _text(shared_core.get("anchor"))
    max_questions = max(0, int(constraints.get("max_questions") or 0))
    brightness = _float01(profile.get("tempo")) if _text(profile.get("brightness")) else 0.0
    playfulness = _float01(profile.get("playfulness"))
    tempo = _float01(profile.get("tempo"))
    voice_texture = _text(profile.get("voice_texture")) or _text(source_state.get("voice_texture"))
    brightness_state = _text(profile.get("brightness"))
    live_state = _text(source_state.get("live_engagement_state"))
    lightness_state = _text(source_state.get("lightness_budget_state"))
    appraisal_state_name = _text(source_state.get("appraisal_state"))
    appraisal_event = _text(source_state.get("appraisal_event")) or _text(profile.get("appraisal_event"))
    appraisal_shared_shift = _text(source_state.get("appraisal_shared_shift")) or _text(profile.get("appraisal_shared_shift"))
    meaning_update_state = _text(source_state.get("meaning_update_state"))
    meaning_update_relation = _text(source_state.get("meaning_update_relation")) or _text(profile.get("meaning_update_relation"))
    utterance_reason_state = _text(source_state.get("utterance_reason_state"))
    utterance_reason_offer = _text(source_state.get("utterance_reason_offer")) or _text(profile.get("utterance_reason_offer"))
    utterance_reason_preserve = _text(source_state.get("utterance_reason_preserve"))
    utterance_reason_question_policy = _text(source_state.get("utterance_reason_question_policy"))
    utterance_reason_tone_hint = _text(source_state.get("utterance_reason_tone_hint"))
    interaction_strategy = _text(source_state.get("interaction_policy_strategy"))
    interaction_opening_move = _text(source_state.get("interaction_policy_opening_move"))
    scene_family = _text(source_state.get("scene_family"))
    actuation_primary_action = _text(source_state.get("actuation_primary_action")) or _text(
        profile.get("actuation_primary_action")
    )
    shared_presence_mode = _text(source_state.get("shared_presence_mode"))
    shared_presence_co_presence = _float01(source_state.get("shared_presence_co_presence"))
    shared_presence_boundary_stability = _float01(
        source_state.get("shared_presence_boundary_stability")
    )
    self_other_dominant_attribution = _text(
        source_state.get("self_other_dominant_attribution")
    )
    self_other_unknown_likelihood = _float01(
        source_state.get("self_other_unknown_likelihood")
    )
    subjective_scene_anchor_frame = _text(source_state.get("subjective_scene_anchor_frame"))
    subjective_scene_shared_scene_potential = _float01(
        source_state.get("subjective_scene_shared_scene_potential")
    )
    organism_posture = _text(profile.get("organism_posture")) or _text(source_state.get("organism_posture"))
    organism_attunement = max(
        _float01(profile.get("organism_attunement")),
        _float01(source_state.get("organism_attunement")),
    )
    organism_grounding = max(
        _float01(profile.get("organism_grounding")),
        _float01(source_state.get("organism_grounding")),
    )
    organism_protective_tension = max(
        _float01(profile.get("organism_protective_tension")),
        _float01(source_state.get("organism_protective_tension")),
    )
    organism_expressive_readiness = max(
        _float01(profile.get("organism_expressive_readiness")),
        _float01(source_state.get("organism_expressive_readiness")),
    )
    organism_play_window = max(
        _float01(profile.get("organism_play_window")),
        _float01(source_state.get("organism_play_window")),
    )
    organism_relation_pull = max(
        _float01(profile.get("organism_relation_pull")),
        _float01(source_state.get("organism_relation_pull")),
    )
    protective_posture = organism_posture in {"protect", "recover", "verify"}
    attuned_posture = organism_posture == "attune"
    playful_posture = organism_posture == "play"
    open_posture = organism_posture == "open"
    concrete_small_event = appraisal_event in {
        "laugh_break",
        "relief_opening",
        "pleasant_turn",
    }
    concrete_shared_shift = (
        appraisal_shared_shift in {
            "shared_smile_window",
            "breathing_room_opened",
            "attention_turns_open",
        }
        or meaning_update_relation in {
            "shared_smile_window",
            "shared_breathing_room",
            "shared_turn_toward_good",
        }
    )
    reason_small_offer = utterance_reason_offer in {
        "brief_shared_smile",
        "brief_relief_ack",
        "brief_good_turn_ack",
    }
    reason_driven_bright = (
        utterance_reason_state == "active"
        and reason_small_offer
        and (concrete_small_event or concrete_shared_shift)
        and appraisal_state_name in {"", "active"}
        and meaning_update_state in {"", "active"}
    )
    shared_self_view_signal = max(
        shared_presence_co_presence,
        subjective_scene_shared_scene_potential,
    )
    guarded_self_view_signal = max(
        self_other_unknown_likelihood,
        max(0.0, 1.0 - shared_presence_boundary_stability),
    )
    self_view_join_ready = (
        self_other_dominant_attribution == "shared"
        and shared_self_view_signal >= 0.48
        and guarded_self_view_signal < 0.55
    )
    self_view_guarded = (
        shared_presence_mode == "guarded_boundary"
        or guarded_self_view_signal >= 0.58
    )
    guarded_repair_context = (
        interaction_strategy in {"repair_then_attune", "respectful_wait"}
        or scene_family in {"repair_window"}
        or interaction_opening_move
        in {"name_overreach_and_reduce_force", "reduce_force_and_secure_boundary"}
        or actuation_primary_action in {"soft_repair"}
    )
    explicit_bright_signal = (
        kind == "bright_continuity"
        or conversation_phase == "bright_continuity"
        or reason_driven_bright
    )
    act_bright_signal = (
        preferred_act in {"shared_delight", "light_bounce"}
        or any(act in {"shared_delight", "light_bounce"} for act in acts)
    )
    bright_signal = explicit_bright_signal or (
        act_bright_signal and not guarded_repair_context
    )
    organism_bright_room = (
        organism_play_window >= 0.34
        and organism_expressive_readiness >= 0.38
        and organism_protective_tension <= 0.64
    )
    bright_room = (
        (reason_driven_bright and (self_view_join_ready or not self_view_guarded))
        or (
            _truthy(constraints.get("allow_small_next_step"))
        and not guarded_repair_context
        and not self_view_guarded
        and (
            bright_signal
            or organism_bright_room
            or voice_texture in {"light_playful", "warm_companion", "gal_bright", "gal_casual"}
            or live_state in {"pickup_comment", "riff_with_comment", "seed_topic"}
            or lightness_state in {"open_play", "warm_only", "light_ok"}
            or brightness_state in {"pickup_comment", "riff_with_comment", "seed_topic", "open_play", "warm_only", "light_ok"}
        )
        )
    )
    bright_allowed = not self_view_guarded or (reason_driven_bright and self_view_join_ready)

    if "reopen_from_anchor" in acts or preferred_act == "reopen_from_anchor":
        return DiscourseShape(
            shape_id="anchor_reopen",
            primary_move="reopen",
            secondary_move="return_point",
            sentence_budget=2,
            question_budget=0,
            anchor_mode="explicit" if anchor else "implicit",
            closing_mode="return_point",
            energy="contained",
            brightness=brightness,
            playfulness=playfulness,
            tempo=tempo,
        )

    if (bright_signal or bright_room) and bright_allowed:
        question_budget = min(max_questions, 1)
        if utterance_reason_question_policy in {"none", "no_question"}:
            question_budget = 0
        elif reason_driven_bright and utterance_reason_preserve == "keep_it_small":
            question_budget = 0
        elif protective_posture:
            question_budget = 0
        bright_playfulness_floor = 0.3
        if utterance_reason_tone_hint == "chatty_ack":
            bright_playfulness_floor = 0.38
        elif utterance_reason_tone_hint == "playful_ack":
            bright_playfulness_floor = 0.46
        if playful_posture:
            bright_playfulness_floor = max(bright_playfulness_floor, 0.52)
        elif open_posture:
            bright_playfulness_floor = max(bright_playfulness_floor, 0.44)
        elif attuned_posture:
            bright_playfulness_floor = max(bright_playfulness_floor, 0.34)
        playfulness_value = max(playfulness, bright_playfulness_floor, organism_play_window)
        if protective_posture and not reason_driven_bright:
            playfulness_value = min(playfulness_value, 0.32)
        brightness_floor = 0.45 if not reason_driven_bright else 0.52
        if playful_posture:
            brightness_floor = max(brightness_floor, 0.6)
        elif open_posture:
            brightness_floor = max(brightness_floor, 0.54)
        elif attuned_posture:
            brightness_floor = max(brightness_floor, 0.5)
        elif protective_posture:
            brightness_floor = max(brightness_floor, 0.46)
        tempo_floor = 0.28 if reason_driven_bright else 0.0
        if playful_posture:
            tempo_floor = max(tempo_floor, 0.38)
        elif open_posture:
            tempo_floor = max(tempo_floor, 0.32)
        elif attuned_posture:
            tempo_floor = max(tempo_floor, 0.24)
        elif protective_posture:
            tempo_floor = max(tempo_floor, 0.16)
        return DiscourseShape(
            shape_id="bright_bounce",
            primary_move="bounce",
            secondary_move="followup" if question_budget > 0 and not protective_posture else "glow",
            sentence_budget=2,
            question_budget=question_budget,
            anchor_mode="implicit" if anchor else "none",
            closing_mode="open_light",
            energy="bright",
            brightness=max(brightness, brightness_floor, organism_relation_pull * 0.5),
            playfulness=playfulness_value,
            tempo=max(tempo, tempo_floor, organism_expressive_readiness * 0.55),
        )

    if self_view_guarded and max_questions == 0:
        return DiscourseShape(
            shape_id="reflect_hold",
            primary_move="reflect",
            secondary_move="stay",
            sentence_budget=2,
            question_budget=0,
            anchor_mode="implicit" if subjective_scene_anchor_frame in {"shared_margin", "front_field"} else "none",
            closing_mode="quiet_presence",
            energy="guarded",
            brightness=brightness,
            playfulness=min(playfulness, 0.18),
            tempo=max(tempo, 0.14),
        )

    if (
        attuned_posture
        and not bright_signal
        and max_questions == 0
        and preferred_act not in {"reopen_from_anchor"}
        and "reopen_from_anchor" not in acts
        and organism_attunement >= 0.54
        and organism_grounding >= 0.34
    ):
        return DiscourseShape(
            shape_id="reflect_hold",
            primary_move="reflect",
            secondary_move="stay",
            sentence_budget=2,
            question_budget=0,
            anchor_mode="none",
            closing_mode="quiet_presence",
            energy="quiet",
            brightness=brightness,
            playfulness=min(playfulness, 0.24),
            tempo=max(tempo, 0.16),
        )

    if (
        kind == "green_reflection_hold"
        or preferred_act in {"reflect_hidden_need", "stay_with_present_need"}
        or ("reflect_hidden_need" in acts and max_questions == 0)
    ):
        return DiscourseShape(
            shape_id="reflect_hold",
            primary_move="reflect",
            secondary_move="stay",
            sentence_budget=2,
            question_budget=0,
            anchor_mode="none",
            closing_mode="quiet_presence",
            energy="quiet",
            brightness=brightness,
            playfulness=playfulness,
            tempo=tempo,
        )

    return DiscourseShape(
        shape_id="reflect_step",
        primary_move="reflect",
        secondary_move="light_question" if max_questions > 0 and not protective_posture else "gentle_close",
        sentence_budget=2,
        question_budget=0 if protective_posture else min(max_questions, 1),
        anchor_mode="implicit" if anchor else "none",
        closing_mode="soft_close" if protective_posture else ("open" if max_questions > 0 else "soft_close"),
        energy="guarded" if protective_posture else "neutral",
        brightness=brightness,
        playfulness=playfulness,
        tempo=tempo,
    )


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return bool(value)
