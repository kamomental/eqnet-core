from __future__ import annotations

from dataclasses import asdict

from ..action_posture import derive_action_posture
from ..actuation_plan import derive_actuation_plan
from ..access_dynamics import advance_access_dynamics
from ..access_projection import project_access_regions
from ..access.models import ForegroundState
from ..affect_blend import derive_affect_blend_state
from ..conscious_workspace import ignite_conscious_workspace
from ..conversation_contract import build_conversation_contract
from ..conversational_objects import derive_conversational_objects
from ..contact_dynamics import advance_contact_dynamics
from ..contact_field import derive_contact_field
from ..constraint_field import derive_constraint_field
from ..interaction_effects import derive_interaction_effects
from ..interaction_audit_bundle import build_interaction_audit_bundle
from ..interaction_audit_casebook import (
    build_interaction_audit_case_entry,
    update_interaction_audit_casebook,
)
from ..interaction_audit_report import build_interaction_audit_report
from ..interaction_condition_report import build_interaction_condition_report
from ..interaction_inspection_report import build_interaction_inspection_report
from ..interaction_judgement_summary import derive_interaction_judgement_summary
from ..interaction_judgement_view import derive_interaction_judgement_view
from ..interaction import (
    compose_nonverbal_profile,
    derive_memory_context_live_regulation,
    derive_relational_mood,
    orchestrate_interaction,
    summarize_situation_state,
)
from ..interaction_option_search import generate_interaction_option_candidates
from ..memory.orchestration import MemoryContext
from ..object_operations import derive_object_operations
from ..partner_style import resolve_partner_utterance_stance
from ..policy_packet import derive_interaction_policy_packet
from ..qualia_kernel_adapter import QualiaPlannerView
from ..resonance_evaluator import (
    evaluate_interaction_resonance,
    rerank_interaction_option_candidates,
)
from ..scene_state import derive_scene_state
from .content_policy import derive_content_sequence, derive_content_skeleton
from .hint_bridge import ensure_qualia_planner_view
from .models import DialogueContext, ResponsePlan
from .surface_language_profile import derive_surface_language_profile
from .surface_profile import derive_surface_profile


def render_response(
    foreground_state: ForegroundState,
    dialogue_context: DialogueContext,
    memory_context: MemoryContext | None = None,
) -> ResponsePlan:
    """LLM には foreground と派生した memory context だけを渡す。"""
    memory_payload: dict[str, object] = {}
    speech_act = "report"
    multimodal_cues = ["steady_voice"] if foreground_state.current_risks else []
    utterance_stance = "neutral_observation"
    nonverbal_profile_payload: dict[str, object] = {}
    # この層は感じを新しく決めるところではなく、
    # 上で決まった感じに合わせて動きや話し方を変えるところに留める。
    # same-turn の shared qualia を正本とし、互換用の補完は hint_bridge に閉じる。
    expression_hints = ensure_qualia_planner_view(
        dialogue_context.expression_hints or {},
    )
    qualia_planner_view_hint = expression_hints.get("qualia_planner_view")
    qualia_state_hint = expression_hints.get("qualia_state")
    qualia_health_hint = expression_hints.get("qualia_estimator_health")
    qualia_protection_grad_hint = expression_hints.get("qualia_protection_grad_x")
    qualia_axis_labels = expression_hints.get("qualia_axis_labels")
    qualia_hint_source = str(expression_hints.get("qualia_hint_source") or "none")
    qualia_hint_version = int(expression_hints.get("qualia_hint_version") or 0)
    qualia_hint_fallback_reason = str(expression_hints.get("qualia_hint_fallback_reason") or "")
    qualia_hint_expected_source = str(expression_hints.get("qualia_hint_expected_source") or "")
    qualia_hint_expected_mismatch = bool(expression_hints.get("qualia_hint_expected_mismatch", False))
    qualia_planner_view = QualiaPlannerView.from_mapping(
        qualia_planner_view_hint if isinstance(qualia_planner_view_hint, dict) else {}
    )
    if memory_context is not None:
        memory_payload = {
            "episodic_candidates": [
                {
                    "episode_id": record.episode_id,
                    "summary": record.summary,
                    "salience": record.salience,
                    "related_person_id": record.related_person_id or None,
                    "fixation_reasons": list(record.fixation_reasons),
                }
                for record in memory_context.episodic_candidates
            ],
            "semantic_hints": [
                {
                    "pattern_id": hint.pattern_id,
                    "label": hint.label,
                    "recurrence_weight": hint.recurrence_weight,
                }
                for hint in memory_context.semantic_hints
            ],
            "continuity_threads": list(memory_context.continuity_threads),
            "retention_summary": list(memory_context.retention_summary),
            "related_person_ids": list(memory_context.related_person_ids),
            "relation_bias_strength": memory_context.relation_bias_strength,
            "partner_semantic_summary": memory_context.partner_semantic_summary,
            "partner_address_hint": memory_context.partner_address_hint,
            "partner_timing_hint": memory_context.partner_timing_hint,
            "partner_stance_hint": memory_context.partner_stance_hint,
        }
        if memory_context.related_person_ids and memory_context.relation_bias_strength >= 0.28:
            speech_act = "check_in"
            utterance_stance = resolve_partner_utterance_stance(
                relation_bias_strength=memory_context.relation_bias_strength,
                related_person_ids=memory_context.related_person_ids,
                partner_address_hint=memory_context.partner_address_hint,
                partner_timing_hint=memory_context.partner_timing_hint,
                partner_stance_hint=memory_context.partner_stance_hint,
            )
            multimodal_cues = sorted(
                set(
                    multimodal_cues
                    + ["gentle_turn_toward_partner"]
                    + (["partner_memory_soft_start"] if memory_context.partner_semantic_summary else [])
                    + ([f"partner_address_{memory_context.partner_address_hint}"] if memory_context.partner_address_hint else [])
                    + ([f"partner_timing_{memory_context.partner_timing_hint}"] if memory_context.partner_timing_hint else [])
                    + ([f"partner_stance_{memory_context.partner_stance_hint}"] if memory_context.partner_stance_hint else [])
                )
            )
    situation_state = summarize_situation_state(
        affective_summary=foreground_state.affective_summary,
        current_risks=foreground_state.current_risks,
        active_goals=foreground_state.active_goals,
        selection_reasons=foreground_state.selection_reasons,
        relation_bias_strength=(memory_context.relation_bias_strength if memory_context is not None else 0.0),
    )
    relational_mood = derive_relational_mood(
        affective_summary=foreground_state.affective_summary,
        situation_state=situation_state,
        partner_address_hint=(memory_context.partner_address_hint if memory_context is not None else ""),
        partner_timing_hint=(memory_context.partner_timing_hint if memory_context is not None else ""),
        partner_stance_hint=(memory_context.partner_stance_hint if memory_context is not None else ""),
    )
    nonverbal_profile = compose_nonverbal_profile(
        utterance_stance=utterance_stance,
        affective_summary=foreground_state.affective_summary,
        situation_state=situation_state,
        relational_mood=relational_mood,
        partner_address_hint=(memory_context.partner_address_hint if memory_context is not None else ""),
        partner_timing_hint=(memory_context.partner_timing_hint if memory_context is not None else ""),
        partner_stance_hint=(memory_context.partner_stance_hint if memory_context is not None else ""),
    )
    live_regulation = derive_memory_context_live_regulation(
        memory_context=memory_context,
        situation_state=situation_state,
        relational_mood=relational_mood,
    )
    scene_state = derive_scene_state(
        place_mode="relational_private" if memory_context and memory_context.related_person_ids else "unspecified",
        privacy_level=_derive_privacy_level(
            relation_bias_strength=(memory_context.relation_bias_strength if memory_context is not None else 0.0),
            social_tension=float(foreground_state.affective_summary.get("social_tension", 0.0) or 0.0),
        ),
        social_topology="one_to_one" if memory_context and memory_context.related_person_ids else "ambient",
        task_phase=_derive_task_phase(foreground_state.active_goals),
        temporal_phase=situation_state.current_phase,
        norm_pressure=_derive_norm_pressure(
            social_tension=float(foreground_state.affective_summary.get("social_tension", 0.0) or 0.0),
            partner_timing_hint=(memory_context.partner_timing_hint if memory_context is not None else ""),
            partner_stance_hint=(memory_context.partner_stance_hint if memory_context is not None else ""),
        ),
        safety_margin=_derive_safety_margin(
            current_risks=foreground_state.current_risks,
            arousal=float(foreground_state.affective_summary.get("arousal", 0.0) or 0.0),
            social_tension=float(foreground_state.affective_summary.get("social_tension", 0.0) or 0.0),
        ),
        environmental_load=_derive_environmental_load(
            arousal=float(foreground_state.affective_summary.get("arousal", 0.0) or 0.0),
            social_tension=float(foreground_state.affective_summary.get("social_tension", 0.0) or 0.0),
            current_risks=foreground_state.current_risks,
        ),
        current_risks=foreground_state.current_risks,
        active_goals=foreground_state.active_goals,
    )
    affect_blend_state = derive_affect_blend_state(
        affective_summary=foreground_state.affective_summary,
        relational_mood=relational_mood,
        live_regulation=live_regulation,
        situation_state=situation_state,
        scene_state=scene_state,
        stress=float(foreground_state.affective_summary.get("stress", 0.0) or 0.0),
        recovery_need=float(foreground_state.affective_summary.get("recovery_need", 0.0) or 0.0),
        safety_bias=float(foreground_state.affective_summary.get("safety_bias", 0.0) or 0.0),
        relation_bias_strength=(memory_context.relation_bias_strength if memory_context is not None else 0.0),
    )
    constraint_field = derive_constraint_field(
        scene_state=scene_state,
        affect_blend=affect_blend_state,
        stress=float(foreground_state.affective_summary.get("stress", 0.0) or 0.0),
        recovery_need=float(foreground_state.affective_summary.get("recovery_need", 0.0) or 0.0),
        safety_bias=float(foreground_state.affective_summary.get("safety_bias", 0.0) or 0.0),
        recent_strain=float(foreground_state.affective_summary.get("social_tension", 0.0) or 0.0),
        current_risks=foreground_state.current_risks,
    )
    contact_field = derive_contact_field(
        affect_blend_state=affect_blend_state.to_dict(),
        constraint_field=constraint_field.to_dict(),
        scene_state=asdict(scene_state),
        current_focus="person" if memory_context and memory_context.related_person_ids else "ambient",
        reportable_facts=foreground_state.reportable_facts,
        current_risks=foreground_state.current_risks,
        related_person_ids=(memory_context.related_person_ids if memory_context is not None else []),
        memory_anchor=(foreground_state.reportable_facts[0] if foreground_state.reportable_facts else ""),
    )
    contact_dynamics = advance_contact_dynamics(
        contact_field=contact_field.to_dict(),
        previous_residue=0.0,
    )
    access_projection = project_access_regions(
        contact_field=contact_field.to_dict(),
        contact_dynamics=contact_dynamics.to_dict(),
        affect_blend_state=affect_blend_state.to_dict(),
        constraint_field=constraint_field.to_dict(),
        qualia_state=qualia_state_hint if isinstance(qualia_state_hint, dict) else None,
        terrain_readout=expression_hints.get("terrain_readout"),
        insight_event=expression_hints.get("insight_event"),
    )
    access_dynamics = advance_access_dynamics(
        access_projection=access_projection.to_dict(),
        previous_residue=0.0,
        current_risks=foreground_state.current_risks,
    )
    interaction_option_candidates = generate_interaction_option_candidates(
        scene_state=scene_state,
        situation_state=situation_state,
        relational_mood=relational_mood,
        live_regulation=live_regulation,
        constraint_field=constraint_field.to_dict(),
    )
    provisional_workspace = ignite_conscious_workspace(
        affect_blend=affect_blend_state,
        constraint_field=constraint_field,
        current_focus="person" if memory_context and memory_context.related_person_ids else "ambient",
        reportable_facts=foreground_state.reportable_facts,
        current_risks=foreground_state.current_risks,
        related_person_ids=(memory_context.related_person_ids if memory_context is not None else []),
        interaction_option_candidates=interaction_option_candidates,
        memory_anchor=(foreground_state.reportable_facts[0] if foreground_state.reportable_facts else ""),
        scene_state=asdict(scene_state),
        contact_field=contact_field.to_dict(),
        contact_dynamics=contact_dynamics.to_dict(),
        access_projection=access_projection.to_dict(),
        access_dynamics=access_dynamics.to_dict(),
    )
    resonance_evaluation = evaluate_interaction_resonance(
        scene_state=scene_state,
        affect_blend=affect_blend_state,
        constraint_field=constraint_field,
        conscious_workspace=provisional_workspace,
        interaction_option_candidates=interaction_option_candidates,
        current_risks=foreground_state.current_risks,
    )
    interaction_option_candidates = rerank_interaction_option_candidates(
        interaction_option_candidates=interaction_option_candidates,
        resonance_evaluation=resonance_evaluation,
    )
    conscious_workspace = ignite_conscious_workspace(
        affect_blend=affect_blend_state,
        constraint_field=constraint_field,
        current_focus="person" if memory_context and memory_context.related_person_ids else "ambient",
        reportable_facts=foreground_state.reportable_facts,
        current_risks=foreground_state.current_risks,
        related_person_ids=(memory_context.related_person_ids if memory_context is not None else []),
        interaction_option_candidates=interaction_option_candidates,
        memory_anchor=(foreground_state.reportable_facts[0] if foreground_state.reportable_facts else ""),
        scene_state=asdict(scene_state),
        contact_field=contact_field.to_dict(),
        contact_dynamics=contact_dynamics.to_dict(),
        access_projection=access_projection.to_dict(),
        access_dynamics=access_dynamics.to_dict(),
        previous_workspace=provisional_workspace.to_dict(),
    )
    relation_context = {
        "relation_bias_strength": (memory_context.relation_bias_strength if memory_context is not None else 0.0),
        "recent_strain": live_regulation.strained_pause,
        "partner_timing_hint": (memory_context.partner_timing_hint if memory_context is not None else ""),
        "partner_stance_hint": (memory_context.partner_stance_hint if memory_context is not None else ""),
        "partner_social_interpretation": (memory_context.partner_semantic_summary if memory_context is not None else ""),
    }
    memory_context_payload = {
        "memory_anchor": (foreground_state.reportable_facts[0] if foreground_state.reportable_facts else ""),
        "relation_seed_summary": (memory_context.partner_semantic_summary if memory_context is not None else ""),
        "long_term_theme_summary": " / ".join(str(item) for item in memory_payload.get("continuity_threads") or [] if str(item).strip()),
        "conscious_residue_summary": " / ".join(str(item) for item in memory_payload.get("retention_summary") or [] if str(item).strip()),
    }
    conversational_objects = derive_conversational_objects(
        current_text=dialogue_context.user_text,
        current_focus="person" if memory_context and memory_context.related_person_ids else "ambient",
        reportable_facts=foreground_state.reportable_facts,
        scene_state=scene_state.__dict__,
        relation_context=relation_context,
        memory_context=memory_context_payload,
        affect_blend_state=affect_blend_state.to_dict(),
        constraint_field=constraint_field.to_dict(),
        conscious_workspace=conscious_workspace.to_dict(),
        resonance_evaluation=resonance_evaluation.to_dict(),
    )
    object_operations = derive_object_operations(
        conversational_objects=conversational_objects.to_dict(),
        scene_state=scene_state.__dict__,
        relation_context=relation_context,
        memory_context=memory_context_payload,
        resonance_evaluation=resonance_evaluation.to_dict(),
        constraint_field=constraint_field.to_dict(),
        conscious_workspace=conscious_workspace.to_dict(),
        interaction_option_candidates=[candidate.__dict__ for candidate in interaction_option_candidates],
    )
    interaction_effects = derive_interaction_effects(
        conversational_objects=conversational_objects.to_dict(),
        object_operations=object_operations.to_dict(),
        resonance_evaluation=resonance_evaluation.to_dict(),
        constraint_field=constraint_field.to_dict(),
    )
    interaction_judgement_view = derive_interaction_judgement_view(
        current_text=dialogue_context.user_text,
        reportable_facts=foreground_state.reportable_facts,
        conversational_objects=conversational_objects.to_dict(),
        object_operations=object_operations.to_dict(),
        interaction_effects=interaction_effects.to_dict(),
        resonance_evaluation=resonance_evaluation.to_dict(),
    )
    interaction_judgement_summary = derive_interaction_judgement_summary(
        interaction_judgement_view=interaction_judgement_view.to_dict(),
        conversational_objects=conversational_objects.to_dict(),
        object_operations=object_operations.to_dict(),
        interaction_effects=interaction_effects.to_dict(),
    )
    interaction_condition_report = build_interaction_condition_report(
        scene_state=scene_state.__dict__,
        resonance_evaluation=resonance_evaluation.to_dict(),
        relation_context=relation_context,
        memory_context=memory_context_payload,
    )
    interaction_inspection_report = build_interaction_inspection_report(
        {"current_case": interaction_judgement_summary.to_dict()}
    )
    interaction_audit_bundle = build_interaction_audit_bundle(
        interaction_judgement_summary=interaction_judgement_summary.to_dict(),
        interaction_condition_report=interaction_condition_report.to_dict(),
        interaction_inspection_report=interaction_inspection_report.to_dict(),
        conversational_objects=conversational_objects.to_dict(),
        object_operations=object_operations.to_dict(),
        interaction_effects=interaction_effects.to_dict(),
        resonance_evaluation=resonance_evaluation.to_dict(),
    )
    interaction_audit_casebook = update_interaction_audit_casebook(
        None,
        build_interaction_audit_case_entry(
            observed_text=dialogue_context.user_text,
            judgement_summary=interaction_judgement_summary.to_dict(),
            audit_bundle=interaction_audit_bundle.to_dict(),
            scene_state=scene_state.__dict__,
            relation_context=relation_context,
            memory_context=memory_context_payload,
        ),
    )
    interaction_audit_report = build_interaction_audit_report(
        {"current_case": interaction_audit_bundle.to_dict()}
    )
    orchestration = orchestrate_interaction(
        current_risks=foreground_state.current_risks,
        situation_state=situation_state,
        relational_mood=relational_mood,
        nonverbal_profile=nonverbal_profile,
        live_regulation=live_regulation,
    )
    surface_profile = derive_surface_profile(
        speech_act=speech_act,
        utterance_stance=utterance_stance,
        orchestration=orchestration,
        live_regulation=live_regulation,
        nonverbal_profile=nonverbal_profile,
    )
    surface_profile = _apply_qualia_planner_surface_bias(surface_profile, qualia_planner_view)
    interaction_policy = derive_interaction_policy_packet(
        dialogue_act=speech_act,
        current_focus="person" if memory_context and memory_context.related_person_ids else "ambient",
        current_risks=foreground_state.current_risks,
        reportable_facts=foreground_state.reportable_facts,
        relation_bias_strength=(memory_context.relation_bias_strength if memory_context is not None else 0.0),
        related_person_ids=(memory_context.related_person_ids if memory_context is not None else []),
        partner_address_hint=(memory_context.partner_address_hint if memory_context is not None else ""),
        partner_timing_hint=(memory_context.partner_timing_hint if memory_context is not None else ""),
        partner_stance_hint=(memory_context.partner_stance_hint if memory_context is not None else ""),
        partner_social_interpretation=(memory_context.partner_semantic_summary if memory_context is not None else ""),
        orchestration=orchestration,
        surface_profile={
            **surface_profile,
            "opening_pace_windowed": surface_profile["opening_delay"],
            "return_gaze_expectation": nonverbal_profile.gaze_mode,
        },
        live_regulation=live_regulation,
        scene_state=scene_state.__dict__,
        interaction_option_candidates=[candidate.__dict__ for candidate in interaction_option_candidates],
        affect_blend_state=affect_blend_state.to_dict(),
        constraint_field=constraint_field.to_dict(),
        conscious_workspace=conscious_workspace.to_dict(),
        resonance_evaluation=resonance_evaluation.to_dict(),
        conversational_objects=conversational_objects.to_dict(),
        object_operations=object_operations.to_dict(),
        interaction_effects=interaction_effects.to_dict(),
        interaction_judgement_view=interaction_judgement_view.to_dict(),
        qualia_planner_view=qualia_planner_view.to_dict(),
        affective_position=expression_hints.get("affective_position"),
        terrain_readout=expression_hints.get("terrain_readout"),
        protection_mode=expression_hints.get("protection_mode"),
        insight_event=expression_hints.get("insight_event"),
        insight_reframing_bias=float(expression_hints.get("insight_reframing_bias") or 0.0),
        insight_class_focus=str(expression_hints.get("insight_class_focus") or "").strip(),
        self_state=foreground_state.affective_summary,
    )
    surface_profile = _apply_interaction_policy_surface_bias(surface_profile, interaction_policy)
    conversation_contract = build_conversation_contract(
        conversational_objects=conversational_objects.to_dict(),
        object_operations=object_operations.to_dict(),
        interaction_effects=interaction_effects.to_dict(),
        interaction_judgement_summary=interaction_judgement_summary.to_dict(),
        interaction_condition_report=interaction_condition_report.to_dict(),
        interaction_policy=interaction_policy,
    )
    interaction_policy["conversation_contract"] = conversation_contract
    content_skeleton = derive_content_skeleton(
        current_text=" ".join(foreground_state.reportable_facts) or "stay with what is visible first",
        interaction_policy=interaction_policy,
        conscious_access={"intent": speech_act},
    )
    content_sequence = derive_content_sequence(
        current_text=" ".join(foreground_state.reportable_facts) or "stay with what is visible first",
        interaction_policy=interaction_policy,
        conscious_access={"intent": speech_act},
    )
    action_posture = derive_action_posture(interaction_policy)
    actuation_plan = derive_actuation_plan(interaction_policy, action_posture)
    multimodal_cues = sorted(set(multimodal_cues + nonverbal_profile.cues))
    multimodal_cues = sorted(set(multimodal_cues + live_regulation.cues))
    multimodal_cues = sorted(set(multimodal_cues + list(orchestration["cues"])))
    multimodal_cues = sorted(set(multimodal_cues + list(surface_profile["cues"])))
    multimodal_cues = sorted(set(multimodal_cues + _derive_qualia_planner_cues(qualia_planner_view)))
    insight_event = dict(expression_hints.get("insight_event") or {})
    if bool(insight_event.get("triggered", False)):
        multimodal_cues = sorted(set(multimodal_cues + ["insight_connection"]))
    primary_operation_kind = str(interaction_policy.get("primary_object_operation", {}).get("operation_kind") or "").strip()
    ordered_effect_kinds = [str(item) for item in interaction_policy.get("ordered_effect_kinds") or [] if str(item).strip()]
    focus_now = str(
        interaction_policy.get("focus_now")
        or conversation_contract.get("focus_now")
        or ""
    ).strip()
    leave_closed_for_now = [
        str(item).strip()
        for item in (
            interaction_policy.get("leave_closed_for_now")
            or conversation_contract.get("leave_closed_for_now")
            or []
        )
        if str(item).strip()
    ]
    if primary_operation_kind:
        multimodal_cues = sorted(set(multimodal_cues + [f"operation_{primary_operation_kind}"]))
    if ordered_effect_kinds:
        multimodal_cues = sorted(set(multimodal_cues + [f"effect_{ordered_effect_kinds[0]}"]))
    if focus_now:
        multimodal_cues = sorted(set(multimodal_cues + ["focus_current_object"]))
    if leave_closed_for_now:
        multimodal_cues = sorted(set(multimodal_cues + ["defer_closed_object"]))
    if not (focus_now or primary_operation_kind or ordered_effect_kinds):
        multimodal_cues = sorted(set(multimodal_cues + [f"policy_{interaction_policy['response_strategy']}"]))
    multimodal_cues = sorted(set(multimodal_cues + [f"scene_{scene_state.scene_family}"]))
    multimodal_cues = sorted(
        set(
            multimodal_cues
            + ([f"option_{interaction_option_candidates[0].family_id}"] if interaction_option_candidates else [])
        )
    )
    multimodal_cues = sorted(set(multimodal_cues + list(affect_blend_state.cues)))
    multimodal_cues = sorted(set(multimodal_cues + list(constraint_field.cues)))
    multimodal_cues = sorted(set(multimodal_cues + list(contact_field.cues)))
    multimodal_cues = sorted(set(multimodal_cues + list(contact_dynamics.cues)))
    multimodal_cues = sorted(set(multimodal_cues + list(access_projection.cues)))
    multimodal_cues = sorted(set(multimodal_cues + list(access_dynamics.cues)))
    multimodal_cues = sorted(set(multimodal_cues + list(conscious_workspace.cues)))
    multimodal_cues = sorted(set(multimodal_cues + list(conversational_objects.cues)))
    multimodal_cues = sorted(set(multimodal_cues + list(object_operations.cues)))
    multimodal_cues = sorted(set(multimodal_cues + list(interaction_effects.cues)))
    multimodal_cues = sorted(set(multimodal_cues + list(interaction_judgement_view.cues)))
    multimodal_cues = sorted(set(multimodal_cues + list(resonance_evaluation.cues)))
    multimodal_cues = sorted(set(multimodal_cues + [f"action_{action_posture['engagement_mode']}"]))
    multimodal_cues = sorted(set(multimodal_cues + [f"actuation_{actuation_plan['primary_action']}"]))
    nonverbal_profile_payload = {
        "gaze_mode": nonverbal_profile.gaze_mode,
        "pause_mode": nonverbal_profile.pause_mode,
        "proximity_mode": nonverbal_profile.proximity_mode,
        "silence_mode": nonverbal_profile.silence_mode,
        "gesture_mode": nonverbal_profile.gesture_mode,
        "scene_mode": situation_state.scene_mode,
        "current_phase": situation_state.current_phase,
        "shared_attention": situation_state.shared_attention,
        "repair_window_open": situation_state.repair_window_open,
        "relational_mood": {
            "future_pull": relational_mood.future_pull,
            "reverence": relational_mood.reverence,
            "innocence": relational_mood.innocence,
            "care": relational_mood.care,
            "shared_world_pull": relational_mood.shared_world_pull,
            "confidence_signal": relational_mood.confidence_signal,
        },
        "live_regulation": {
            "past_loop_pull": live_regulation.past_loop_pull,
            "future_loop_pull": live_regulation.future_loop_pull,
            "fantasy_loop_pull": live_regulation.fantasy_loop_pull,
            "shared_attention_active": live_regulation.shared_attention_active,
            "strained_pause": live_regulation.strained_pause,
            "repair_window_open": live_regulation.repair_window_open,
            "distance_expectation": live_regulation.distance_expectation,
        },
        "interaction_orchestration": dict(orchestration),
        "scene_state": scene_state.__dict__,
        "interaction_option_candidates": [candidate.__dict__ for candidate in interaction_option_candidates],
        "contact_field": contact_field.to_dict(),
        "contact_dynamics": contact_dynamics.to_dict(),
        "access_projection": access_projection.to_dict(),
        "access_dynamics": access_dynamics.to_dict(),
        "affect_blend_state": affect_blend_state.to_dict(),
        "constraint_field": constraint_field.to_dict(),
        "conscious_workspace": conscious_workspace.to_dict(),
        "conversational_objects": conversational_objects.to_dict(),
        "object_operations": object_operations.to_dict(),
        "interaction_effects": interaction_effects.to_dict(),
        "conversation_contract": conversation_contract,
        "interaction_judgement_view": interaction_judgement_view.to_dict(),
        "interaction_judgement_summary": interaction_judgement_summary.to_dict(),
        "interaction_condition_report": interaction_condition_report.to_dict(),
        "interaction_inspection_report": interaction_inspection_report.to_dict(),
        "interaction_audit_bundle": interaction_audit_bundle.to_dict(),
        "interaction_audit_casebook": interaction_audit_casebook,
        "interaction_audit_report": interaction_audit_report.to_dict(),
        "interaction_audit_reference_case_ids": [],
        "interaction_audit_reference_case_meta": {},
        "qualia_hint_source": qualia_hint_source,
        "qualia_hint_version": qualia_hint_version,
        "qualia_hint_fallback_reason": qualia_hint_fallback_reason,
        "qualia_hint_expected_source": qualia_hint_expected_source,
        "qualia_hint_expected_mismatch": qualia_hint_expected_mismatch,
        "resonance_evaluation": resonance_evaluation.to_dict(),
        "qualia_planner_view": qualia_planner_view.to_dict(),
    }
    return ResponsePlan(
        speech_act=speech_act,
        content_brief=list(foreground_state.reportable_facts),
        multimodal_cues=multimodal_cues,
        interaction_policy=interaction_policy,
        action_posture=action_posture,
        actuation_plan=actuation_plan,
        surface_profile=surface_profile,
        llm_payload={
            "foreground": {
                "salient_entities": list(foreground_state.salient_entities),
                "current_risks": list(foreground_state.current_risks),
                "active_goals": list(foreground_state.active_goals),
                "affective_summary": dict(foreground_state.affective_summary),
                "reportable_facts": list(foreground_state.reportable_facts),
                "reportability_scores": dict(foreground_state.reportability_scores),
                "memory_candidates": list(foreground_state.memory_candidates),
                "uncertainty_notes": list(foreground_state.uncertainty_notes),
                "selection_reasons": list(foreground_state.selection_reasons),
            },
            "memory_context": memory_payload,
            "utterance_stance": utterance_stance,
            "interaction_policy": interaction_policy,
            "action_posture": action_posture,
            "actuation_plan": actuation_plan,
            "content_skeleton": content_skeleton,
            "content_sequence": content_sequence,
            "surface_profile": surface_profile,
            "nonverbal_profile": nonverbal_profile_payload,
            "scene_state": scene_state.__dict__,
            "interaction_option_candidates": [candidate.__dict__ for candidate in interaction_option_candidates],
            "contact_field": contact_field.to_dict(),
            "contact_dynamics": contact_dynamics.to_dict(),
            "access_projection": access_projection.to_dict(),
            "access_dynamics": access_dynamics.to_dict(),
            "affect_blend_state": affect_blend_state.to_dict(),
            "constraint_field": constraint_field.to_dict(),
            "conscious_workspace": conscious_workspace.to_dict(),
            "conversational_objects": conversational_objects.to_dict(),
            "object_operations": object_operations.to_dict(),
            "interaction_effects": interaction_effects.to_dict(),
            "conversation_contract": conversation_contract,
            "qualia_state": qualia_state_hint if isinstance(qualia_state_hint, dict) else {},
            "qualia_estimator_health": qualia_health_hint if isinstance(qualia_health_hint, dict) else {},
            "qualia_protection_grad_x": qualia_protection_grad_hint if isinstance(qualia_protection_grad_hint, list) else [],
            "qualia_axis_labels": qualia_axis_labels if isinstance(qualia_axis_labels, list) else [],
            "qualia_hint_source": qualia_hint_source,
            "qualia_hint_version": qualia_hint_version,
            "qualia_hint_fallback_reason": qualia_hint_fallback_reason,
            "qualia_hint_expected_source": qualia_hint_expected_source,
            "qualia_hint_expected_mismatch": qualia_hint_expected_mismatch,
            "qualia_planner_view": qualia_planner_view.to_dict(),
            "interaction_judgement_view": interaction_judgement_view.to_dict(),
            "interaction_judgement_summary": interaction_judgement_summary.to_dict(),
            "interaction_condition_report": interaction_condition_report.to_dict(),
            "interaction_inspection_report": interaction_inspection_report.to_dict(),
            "interaction_audit_bundle": interaction_audit_bundle.to_dict(),
            "interaction_audit_casebook": interaction_audit_casebook,
            "interaction_audit_report": interaction_audit_report.to_dict(),
            "interaction_audit_reference_case_ids": [],
            "interaction_audit_reference_case_meta": {},
            "resonance_evaluation": resonance_evaluation.to_dict(),
            "dialogue_context": {
                "user_text": dialogue_context.user_text,
                "history": list(dialogue_context.history),
            },
        },
    )


def _apply_qualia_planner_surface_bias(
    surface_profile: dict[str, object],
    qualia_planner_view: QualiaPlannerView,
) -> dict[str, object]:
    if (
        qualia_planner_view.dominant_axis is None
        and qualia_planner_view.felt_energy <= 0.0
        and qualia_planner_view.protection_bias <= 0.0
        and not qualia_planner_view.degraded
    ):
        return surface_profile

    updated = dict(surface_profile)
    opening_delay = str(updated.get("opening_delay") or "brief")
    response_length = str(updated.get("response_length") or "balanced")
    sentence_temperature = str(updated.get("sentence_temperature") or "neutral")
    pause_insertion = str(updated.get("pause_insertion") or "none")
    certainty_style = str(updated.get("certainty_style") or "direct")

    if qualia_planner_view.degraded or qualia_planner_view.trust <= 0.45:
        if opening_delay == "brief":
            opening_delay = "measured"
        if pause_insertion == "none":
            pause_insertion = "soft_pause"
        certainty_style = "careful"

    if qualia_planner_view.protection_bias >= 0.12 and sentence_temperature == "neutral":
        sentence_temperature = "gentle"

    if (
        qualia_planner_view.felt_energy >= 0.18
        and response_length == "short"
        and not qualia_planner_view.degraded
    ):
        response_length = "balanced"

    cues: list[str] = [
        f"surface_delay_{opening_delay}",
        f"surface_length_{response_length}",
        f"surface_temperature_{sentence_temperature}",
        f"surface_certainty_{certainty_style}",
    ]
    if pause_insertion != "none":
        cues.append(f"surface_pause_{pause_insertion}")

    updated["opening_delay"] = opening_delay
    updated["response_length"] = response_length
    updated["sentence_temperature"] = sentence_temperature
    updated["pause_insertion"] = pause_insertion
    updated["certainty_style"] = certainty_style
    updated["cues"] = cues
    return updated


def _apply_interaction_policy_surface_bias(
    surface_profile: dict[str, object],
    interaction_policy: dict[str, object],
) -> dict[str, object]:
    updated = dict(surface_profile)
    body_recovery_guard = dict(interaction_policy.get("body_recovery_guard") or {})
    body_homeostasis_state = dict(interaction_policy.get("body_homeostasis_state") or {})
    homeostasis_budget_state = dict(interaction_policy.get("homeostasis_budget_state") or {})
    initiative_readiness = dict(interaction_policy.get("initiative_readiness") or {})
    agenda_window_state = dict(interaction_policy.get("agenda_window_state") or {})
    commitment_state = dict(interaction_policy.get("commitment_state") or {})
    attention_regulation_state = dict(interaction_policy.get("attention_regulation_state") or {})
    grice_guard_state = dict(interaction_policy.get("grice_guard_state") or {})
    relational_style_memory_state = dict(interaction_policy.get("relational_style_memory_state") or {})
    cultural_conversation_state = dict(interaction_policy.get("cultural_conversation_state") or {})
    expressive_style_state = dict(interaction_policy.get("expressive_style_state") or {})
    lightness_budget_state = dict(interaction_policy.get("lightness_budget_state") or {})
    relational_continuity_state = dict(interaction_policy.get("relational_continuity_state") or {})
    social_topology_state = dict(interaction_policy.get("social_topology_state") or {})
    protection_mode = dict(interaction_policy.get("protection_mode") or {})

    recovery_state = str(body_recovery_guard.get("state") or "").strip()
    recovery_score = _clamp01(float(body_recovery_guard.get("score", 0.0) or 0.0))
    body_homeostasis_name = str(body_homeostasis_state.get("state") or "").strip()
    body_homeostasis_score = _clamp01(float(body_homeostasis_state.get("score", 0.0) or 0.0))
    homeostasis_budget_name = str(homeostasis_budget_state.get("state") or "").strip()
    homeostasis_budget_score = _clamp01(float(homeostasis_budget_state.get("score", 0.0) or 0.0))
    initiative_state = str(initiative_readiness.get("state") or "").strip()
    initiative_score = _clamp01(float(initiative_readiness.get("score", 0.0) or 0.0))
    agenda_window_name = str(agenda_window_state.get("state") or "").strip()
    agenda_window_deferral_budget = _clamp01(float(agenda_window_state.get("deferral_budget", 0.0) or 0.0))
    agenda_window_carry_target = str(agenda_window_state.get("carry_target") or "").strip()
    commitment_mode = str(commitment_state.get("state") or "").strip()
    commitment_target = str(commitment_state.get("target") or "").strip()
    commitment_score = _clamp01(float(commitment_state.get("score", 0.0) or 0.0))
    attention_state = str(attention_regulation_state.get("state") or "").strip()
    grice_state = str(grice_guard_state.get("state") or "").strip()
    relational_style_name = str(relational_style_memory_state.get("state") or "").strip()
    relational_playful_ceiling = _clamp01(float(relational_style_memory_state.get("playful_ceiling", 0.0) or 0.0))
    relational_advice_tolerance = _clamp01(float(relational_style_memory_state.get("advice_tolerance", 0.0) or 0.0))
    relational_lexical_variation_bias = _clamp01(float(relational_style_memory_state.get("lexical_variation_bias", 0.0) or 0.0))
    relational_banter_room = _clamp01(float(relational_style_memory_state.get("banter_room", 0.0) or 0.0))
    relational_banter_style = str(relational_style_memory_state.get("banter_style") or "grounded_companion").strip() or "grounded_companion"
    cultural_state_name = str(cultural_conversation_state.get("state") or "").strip()
    cultural_directness_ceiling = _clamp01(float(cultural_conversation_state.get("directness_ceiling", 0.0) or 0.0))
    cultural_joke_ratio_ceiling = _clamp01(float(cultural_conversation_state.get("joke_ratio_ceiling", 0.0) or 0.0))
    cultural_politeness_pressure = _clamp01(float(cultural_conversation_state.get("politeness_pressure", 0.0) or 0.0))
    expressive_style_name = str(expressive_style_state.get("state") or "").strip()
    expressive_style_score = _clamp01(float(expressive_style_state.get("score", 0.0) or 0.0))
    expressive_lightness_room = _clamp01(float(expressive_style_state.get("lightness_room", 0.0) or 0.0))
    expressive_continuity_weight = _clamp01(float(expressive_style_state.get("continuity_weight", 0.0) or 0.0))
    expressive_style_history_focus = str(interaction_policy.get("expressive_style_history_focus") or "").strip()
    expressive_style_history_bias = _clamp01(float(interaction_policy.get("expressive_style_history_bias", 0.0) or 0.0))
    banter_style_focus = str(interaction_policy.get("banter_style_focus") or "").strip()
    lexical_variation_carry_bias = _clamp01(float(interaction_policy.get("lexical_variation_carry_bias", 0.0) or 0.0))
    lightness_budget_name = str(lightness_budget_state.get("state") or "").strip()
    lightness_banter_room = _clamp01(float(lightness_budget_state.get("banter_room", 0.0) or 0.0))
    lightness_playful_ceiling = _clamp01(float(lightness_budget_state.get("playful_ceiling", 0.0) or 0.0))
    lightness_suppression = _clamp01(float(lightness_budget_state.get("suppression", 0.0) or 0.0))
    relational_continuity_name = str(relational_continuity_state.get("state") or "").strip()
    relational_continuity_score = _clamp01(float(relational_continuity_state.get("score", 0.0) or 0.0))
    social_topology_name = str(social_topology_state.get("state") or "").strip()
    social_topology_score = _clamp01(float(social_topology_state.get("score", 0.0) or 0.0))
    protection_mode_name = str(protection_mode.get("mode") or "").strip()
    forward_warm_safe = (
        recovery_state == "open"
        and body_homeostasis_name not in {"recovering", "depleted"}
        and homeostasis_budget_name != "depleted"
        and protection_mode_name not in {"contain", "stabilize", "shield"}
        and grice_state != "hold_obvious_advice"
        and social_topology_name not in {"public_visible", "hierarchical"}
    )
    forward_warm_room = (
        expressive_lightness_room >= 0.28
        or expressive_style_name in {"warm_companion", "light_playful"}
        or relational_continuity_name == "co_regulating"
    )

    opening_delay = str(updated.get("opening_delay") or "brief")
    response_length = str(updated.get("response_length") or "balanced")
    sentence_temperature = str(updated.get("sentence_temperature") or "neutral")
    pause_insertion = str(updated.get("pause_insertion") or "none")
    certainty_style = str(updated.get("certainty_style") or "direct")
    cues = list(updated.get("cues") or [])

    if recovery_state == "recovery_first":
        if opening_delay in {"brief", "measured"}:
            opening_delay = "long"
        response_length = "short"
        sentence_temperature = "measured" if sentence_temperature == "neutral" else sentence_temperature
        pause_insertion = "soft_pause"
        certainty_style = "careful"
        cues.extend(
            [
                "surface_recovery_first",
                "surface_restore_body_margin",
            ]
        )
    elif recovery_state == "guarded":
        if opening_delay == "brief":
            opening_delay = "measured"
        if response_length == "forward_leaning":
            response_length = "balanced"
        if pause_insertion == "none":
            pause_insertion = "soft_pause"
        if certainty_style == "direct":
            certainty_style = "tentative"
        cues.append("surface_guarded_recovery")

    if body_homeostasis_name == "depleted":
        opening_delay = "long"
        response_length = "short"
        sentence_temperature = "measured"
        pause_insertion = "soft_pause"
        certainty_style = "careful"
        cues.extend(["surface_homeostasis_depleted", "surface_reduce_load"])
    elif body_homeostasis_name == "recovering" and body_homeostasis_score >= 0.32:
        if opening_delay == "brief":
            opening_delay = "measured"
        if response_length == "forward_leaning":
            response_length = "balanced"
        if sentence_temperature == "neutral":
            sentence_temperature = "gentle"
        if pause_insertion == "none":
            pause_insertion = "soft_pause"
        if certainty_style == "direct":
            certainty_style = "tentative"
        cues.append("surface_homeostasis_recovering")
    elif body_homeostasis_name == "strained" and opening_delay == "brief":
        opening_delay = "measured"

    if homeostasis_budget_name == "depleted" and homeostasis_budget_score >= 0.3:
        if opening_delay == "brief":
            opening_delay = "measured"
        if response_length == "forward_leaning":
            response_length = "balanced"
        if sentence_temperature == "warm":
            sentence_temperature = "gentle"
        cues.append("surface_homeostasis_budget_depleted")
    elif homeostasis_budget_name == "recovering" and homeostasis_budget_score >= 0.32:
        if response_length == "forward_leaning":
            response_length = "balanced"
        if pause_insertion == "none":
            pause_insertion = "soft_pause"
        cues.append("surface_homeostasis_budget_recovering")

    if relational_continuity_name == "holding_thread":
        if response_length == "forward_leaning":
            response_length = "balanced"
        if sentence_temperature == "neutral":
            sentence_temperature = "gentle"
        if pause_insertion == "none":
            pause_insertion = "soft_pause"
        cues.extend(["surface_relational_holding", "surface_leave_return_point"])
    elif relational_continuity_name == "reopening":
        if sentence_temperature in {"neutral", "measured"}:
            sentence_temperature = "gentle"
        if certainty_style == "direct":
            certainty_style = "tentative"
        if pause_insertion == "none":
            pause_insertion = "soft_pause"
        cues.append("surface_relational_reopening")
    elif (
        relational_continuity_name == "co_regulating"
        and relational_continuity_score >= 0.38
        and body_homeostasis_name not in {"recovering", "depleted"}
    ):
        if response_length == "short":
            response_length = "balanced"
        if sentence_temperature in {"neutral", "measured"}:
            sentence_temperature = "warm"
        cues.append("surface_relational_coregulating")

    if agenda_window_name == "next_private_window":
        if response_length == "forward_leaning":
            response_length = "balanced"
        if certainty_style == "direct":
            certainty_style = "tentative"
        if pause_insertion == "none":
            pause_insertion = "soft_pause"
        cues.extend(["surface_wait_private_window", "surface_leave_return_point"])
    elif agenda_window_name == "next_same_group_window":
        if response_length == "forward_leaning":
            response_length = "balanced"
        if sentence_temperature == "warm":
            sentence_temperature = "gentle"
        cues.extend(["surface_wait_group_thread", "surface_keep_group_threads_visible"])
    elif agenda_window_name == "next_same_culture_window":
        if certainty_style == "direct":
            certainty_style = "tentative"
        if sentence_temperature == "warm":
            sentence_temperature = "gentle"
        cues.extend(["surface_wait_same_culture_window", "surface_hold_cultural_thread"])
    elif agenda_window_name == "opportunistic_reentry":
        if sentence_temperature == "neutral":
            sentence_temperature = "gentle"
        cues.append("surface_soft_opportunistic_reentry")
    elif agenda_window_name == "long_hold":
        if response_length == "forward_leaning":
            response_length = "short"
        if certainty_style == "direct":
            certainty_style = "careful"
        cues.append("surface_long_hold")

    if social_topology_name == "public_visible" and social_topology_score >= 0.34:
        if opening_delay == "brief":
            opening_delay = "measured"
        if response_length == "forward_leaning":
            response_length = "balanced"
        if certainty_style == "direct":
            certainty_style = "tentative"
        cues.extend(["surface_public_visible", "surface_keep_visibility_safe"])
    elif social_topology_name == "hierarchical" and social_topology_score >= 0.34:
        if opening_delay == "brief":
            opening_delay = "measured"
        if certainty_style == "direct":
            certainty_style = "careful"
        cues.extend(["surface_hierarchical", "surface_respect_role_gradient"])
    elif social_topology_name == "threaded_group" and social_topology_score >= 0.34:
        if response_length == "forward_leaning":
            response_length = "balanced"
        if pause_insertion == "none":
            pause_insertion = "soft_pause"
        cues.append("surface_threaded_group")

    if cultural_state_name == "public_courteous":
        if response_length == "forward_leaning":
            response_length = "balanced"
        if sentence_temperature == "warm":
            sentence_temperature = "gentle"
        if certainty_style == "direct":
            certainty_style = "tentative"
        cues.extend(["surface_cultural_public_courteous", "surface_group_safe_register"])
    elif cultural_state_name == "hierarchy_respectful":
        if response_length == "forward_leaning":
            response_length = "balanced"
        if sentence_temperature in {"warm", "gentle"}:
            sentence_temperature = "measured"
        if certainty_style in {"direct", "tentative"}:
            certainty_style = "careful"
        cues.extend(["surface_cultural_hierarchy_respectful", "surface_keep_role_respect"])
    elif cultural_state_name == "group_attuned":
        if response_length == "short":
            response_length = "balanced"
        if pause_insertion == "none":
            pause_insertion = "soft_pause"
        if sentence_temperature == "neutral" and cultural_joke_ratio_ceiling >= 0.26:
            sentence_temperature = "gentle"
        cues.append("surface_cultural_group_attuned")
    elif cultural_state_name == "casual_shared" and cultural_joke_ratio_ceiling >= 0.34 and lightness_budget_name == "open_play":
        if sentence_temperature in {"neutral", "measured", "gentle"}:
            sentence_temperature = "warm"
        cues.append("surface_cultural_casual_shared")

    if relational_banter_style == "respectful_light":
        if sentence_temperature == "warm":
            sentence_temperature = "gentle"
        if certainty_style == "direct":
            certainty_style = "tentative"
        cues.append("surface_banter_respectful_light")
    elif relational_banter_style == "soft_formal":
        if sentence_temperature in {"neutral", "warm"}:
            sentence_temperature = "gentle"
        cues.append("surface_banter_soft_formal")
    elif (
        relational_banter_style in {"gentle_tease", "compact_wit"}
        and lightness_budget_name == "open_play"
        and grice_state not in {"hold_obvious_advice", "attune_without_repeating"}
        and social_topology_name not in {"public_visible", "hierarchical"}
        and recovery_state == "open"
    ):
        if sentence_temperature in {"neutral", "measured", "gentle"}:
            sentence_temperature = "warm"
        if response_length == "short":
            response_length = "balanced"
        cues.append(f"surface_banter_{relational_banter_style}")
    elif relational_banter_style == "warm_refrain":
        if sentence_temperature in {"neutral", "measured"}:
            sentence_temperature = "warm"
        cues.append("surface_banter_warm_refrain")

    if expressive_style_history_focus:
        cues.append(f"surface_style_history_{expressive_style_history_focus}")
        if expressive_style_history_bias >= 0.08 and expressive_style_history_focus == "warm_companion":
            if sentence_temperature in {"neutral", "measured", "gentle"}:
                sentence_temperature = "warm"
        elif expressive_style_history_bias >= 0.08 and expressive_style_history_focus == "quiet_repair":
            if sentence_temperature in {"neutral", "measured"}:
                sentence_temperature = "gentle"
    if banter_style_focus and lexical_variation_carry_bias >= 0.08:
        cues.append(f"surface_banter_history_{banter_style_focus}")
    if (
        relational_lexical_variation_bias >= 0.34
        or lexical_variation_carry_bias >= 0.1
    ) and lightness_budget_name in {"open_play", "warm_only"}:
        cues.append("surface_lexical_variation_open")

    if attention_state == "reflex_guard":
        if opening_delay == "brief":
            opening_delay = "measured"
        if response_length == "forward_leaning":
            response_length = "balanced"
        if certainty_style == "direct":
            certainty_style = "careful"
        cues.append("surface_reflex_attention_guard")
    elif attention_state == "split_guarded":
        if response_length == "forward_leaning":
            response_length = "balanced"
        if pause_insertion == "none":
            pause_insertion = "soft_pause"
        if certainty_style == "direct":
            certainty_style = "tentative"
        cues.append("surface_split_attention_guard")

    if grice_state == "hold_obvious_advice":
        response_length = "short"
        certainty_style = "careful"
        if sentence_temperature == "warm":
            sentence_temperature = "gentle"
        if pause_insertion == "none":
            pause_insertion = "soft_pause"
        cues.extend(["surface_grice_hold_obvious_advice", "surface_reduce_redundant_advice"])
    elif grice_state == "attune_without_repeating":
        if response_length == "forward_leaning":
            response_length = "balanced"
        if certainty_style == "direct":
            certainty_style = "tentative"
        if pause_insertion == "none":
            pause_insertion = "soft_pause"
        cues.append("surface_grice_attune_without_repeating")
    elif grice_state == "acknowledge_then_extend":
        if response_length == "forward_leaning":
            response_length = "balanced"
        cues.append("surface_grice_acknowledge_then_extend")

    if expressive_style_name == "reverent_measured" and expressive_style_score >= 0.3:
        if opening_delay == "brief":
            opening_delay = "measured"
        if response_length == "forward_leaning":
            response_length = "balanced"
        if sentence_temperature == "warm":
            sentence_temperature = "measured"
        elif sentence_temperature == "neutral":
            sentence_temperature = "gentle"
        if certainty_style == "direct":
            certainty_style = "careful"
        cues.append("surface_expressive_reverent_measured")
    elif expressive_style_name == "quiet_repair" and expressive_style_score >= 0.3:
        if response_length == "short":
            response_length = "balanced"
        if sentence_temperature in {"neutral", "measured"}:
            sentence_temperature = "gentle"
        if pause_insertion == "none":
            pause_insertion = "soft_pause"
        if certainty_style == "direct":
            certainty_style = "tentative"
        cues.append("surface_expressive_quiet_repair")
    elif expressive_style_name == "warm_companion" and expressive_style_score >= 0.3:
        if response_length == "short" and grice_state != "hold_obvious_advice":
            response_length = "balanced"
        if sentence_temperature in {"neutral", "measured", "gentle"}:
            sentence_temperature = "warm"
        if lightness_budget_name == "suppress_play" and sentence_temperature == "warm":
            sentence_temperature = "gentle"
        if certainty_style == "careful" and recovery_state == "open":
            certainty_style = "tentative"
        cues.append("surface_expressive_warm_companion")
    elif (
        expressive_style_name == "light_playful"
        and expressive_style_score >= 0.34
        and expressive_lightness_room >= 0.4
        and lightness_banter_room >= 0.34
        and lightness_playful_ceiling >= 0.3
        and recovery_state == "open"
        and protection_mode_name not in {"contain", "stabilize", "shield"}
        and grice_state not in {"hold_obvious_advice", "attune_without_repeating"}
        and lightness_budget_name == "open_play"
    ):
        if opening_delay == "long":
            opening_delay = "measured"
        if response_length == "short":
            response_length = "balanced"
        if sentence_temperature in {"neutral", "measured", "gentle"}:
            sentence_temperature = "warm"
        if certainty_style == "careful":
            certainty_style = "tentative"
        cues.append("surface_expressive_light_playful")
    elif expressive_style_name == "grounded_gentle" and expressive_style_score >= 0.28:
        if sentence_temperature == "neutral":
            sentence_temperature = "gentle"
        cues.append("surface_expressive_grounded_gentle")

    if lightness_budget_name == "suppress_play":
        if sentence_temperature == "warm":
            sentence_temperature = "gentle"
        if response_length == "forward_leaning":
            response_length = "balanced"
        if pause_insertion == "none":
            pause_insertion = "soft_pause"
        cues.append("surface_lightness_suppress_play")
    elif lightness_budget_name == "grounded_only":
        if sentence_temperature == "warm" and expressive_style_name == "light_playful":
            sentence_temperature = "gentle"
        if response_length == "forward_leaning" and expressive_style_name == "light_playful":
            response_length = "balanced"
        cues.append("surface_lightness_grounded_only")
    elif lightness_budget_name == "warm_only":
        if expressive_style_name == "light_playful" and sentence_temperature in {"neutral", "measured", "gentle"}:
            sentence_temperature = "warm"
        cues.append("surface_lightness_warm_only")
    elif lightness_budget_name == "open_play":
        if (
            expressive_style_name in {"light_playful", "warm_companion"}
            and sentence_temperature in {"neutral", "measured", "gentle"}
            and lightness_banter_room >= 0.34
            and lightness_suppression < 0.32
        ):
            sentence_temperature = "warm"
        if response_length == "short" and relational_banter_room >= 0.32 and relational_playful_ceiling >= 0.3:
            response_length = "balanced"
        cues.append("surface_lightness_open_play")

    if (
        initiative_state == "ready"
        and initiative_score >= 0.32
        and recovery_state == "open"
        and protection_mode_name not in {"contain", "stabilize", "shield"}
    ):
        if opening_delay == "long":
            opening_delay = "measured"
        if response_length in {"short", "balanced"}:
            response_length = "forward_leaning"
        if sentence_temperature in {"neutral", "measured"} or (
            sentence_temperature == "gentle" and forward_warm_safe and forward_warm_room
        ):
            sentence_temperature = "warm"
        if certainty_style == "careful":
            certainty_style = "tentative"
        cues.extend(
            [
                "surface_initiative_ready",
                "surface_offer_next_step",
            ]
        )
    elif initiative_state == "tentative" and recovery_state != "recovery_first":
        if response_length == "short":
            response_length = "balanced"
        if sentence_temperature == "neutral":
            sentence_temperature = "gentle"
        cues.append("surface_initiative_tentative")

    if commitment_mode == "commit":
        if commitment_target == "step_forward" and recovery_state == "open":
            if response_length in {"short", "balanced"}:
                response_length = "forward_leaning"
            if certainty_style in {"careful", "tentative"}:
                certainty_style = "steady"
            if sentence_temperature in {"neutral", "measured"} or (
                sentence_temperature == "gentle" and forward_warm_safe and forward_warm_room
            ):
                sentence_temperature = "warm"
            cues.append("surface_commit_step_forward")
        elif commitment_target in {"repair", "bond_protect"}:
            if certainty_style in {"careful", "tentative"}:
                certainty_style = "steady"
            if sentence_temperature in {"neutral", "measured"}:
                sentence_temperature = "gentle"
            if pause_insertion == "none":
                pause_insertion = "soft_pause"
            cues.append(f"surface_commit_{commitment_target}")
        elif commitment_target in {"stabilize", "hold"}:
            if opening_delay == "brief":
                opening_delay = "measured"
            if certainty_style == "direct":
                certainty_style = "steady"
            cues.append(f"surface_commit_{commitment_target}")
    elif commitment_mode == "waver" and commitment_score >= 0.2:
        if certainty_style == "direct":
            certainty_style = "careful"
        if pause_insertion == "none":
            pause_insertion = "soft_pause"
        cues.append("surface_commitment_waver")

    if grice_state == "hold_obvious_advice":
        certainty_style = "careful"
    if cultural_directness_ceiling <= 0.36 and certainty_style == "direct":
        certainty_style = "tentative"
    if cultural_politeness_pressure >= 0.32 and sentence_temperature == "warm" and lightness_budget_name != "open_play":
        sentence_temperature = "gentle"

    surface_language_profile = derive_surface_language_profile(
        recovery_state=recovery_state,
        protection_mode_name=protection_mode_name,
        grice_state=grice_state,
        expressive_style_name=expressive_style_name,
        expressive_style_history_focus=expressive_style_history_focus,
        relational_continuity_name=relational_continuity_name,
        relational_banter_style=relational_banter_style,
        relational_lexical_variation_bias=relational_lexical_variation_bias,
        relational_banter_room=relational_banter_room,
        lightness_budget_name=lightness_budget_name,
        lightness_banter_room=lightness_banter_room,
        lightness_playful_ceiling=lightness_playful_ceiling,
        lightness_suppression=lightness_suppression,
        social_topology_name=social_topology_name,
        cultural_state_name=cultural_state_name,
        cultural_joke_ratio_ceiling=cultural_joke_ratio_ceiling,
        lexical_variation_carry_bias=lexical_variation_carry_bias,
    )
    if surface_language_profile.banter_move != "none":
        cues.append(f"surface_language_banter_{surface_language_profile.banter_move}")
    if surface_language_profile.lexical_variation_mode != "plain":
        cues.append(
            f"surface_language_lexical_{surface_language_profile.lexical_variation_mode}"
        )
    if surface_language_profile.group_register:
        cues.append(f"surface_language_group_{surface_language_profile.group_register}")

    updated["opening_delay"] = opening_delay
    updated["response_length"] = response_length
    updated["sentence_temperature"] = sentence_temperature
    updated["pause_insertion"] = pause_insertion
    updated["certainty_style"] = certainty_style
    updated["voice_texture"] = expressive_style_name or "grounded_gentle"
    updated["lightness_room"] = round(expressive_lightness_room, 4)
    updated["continuity_weight"] = round(expressive_continuity_weight, 4)
    updated["relational_voice_texture"] = relational_style_name or "grounded_gentle"
    updated["relational_banter_style"] = relational_banter_style
    updated["relational_playful_ceiling"] = round(relational_playful_ceiling, 4)
    updated["relational_advice_tolerance"] = round(relational_advice_tolerance, 4)
    updated["relational_lexical_variation_bias"] = round(relational_lexical_variation_bias, 4)
    updated["cultural_register"] = cultural_state_name or "careful_polite"
    updated["cultural_directness_ceiling"] = round(cultural_directness_ceiling, 4)
    updated["cultural_joke_ratio_ceiling"] = round(cultural_joke_ratio_ceiling, 4)
    updated["lightness_budget_state"] = lightness_budget_name or "grounded_only"
    updated["lightness_banter_room"] = round(lightness_banter_room, 4)
    updated["lightness_suppression"] = round(lightness_suppression, 4)
    updated["agenda_window_state"] = agenda_window_name or "long_hold"
    updated["agenda_window_deferral_budget"] = round(agenda_window_deferral_budget, 4)
    updated["agenda_window_carry_target"] = agenda_window_carry_target or "later_safe_window"
    updated["expressive_style_history_focus"] = expressive_style_history_focus
    updated["expressive_style_history_bias"] = round(expressive_style_history_bias, 4)
    updated["banter_style_focus"] = banter_style_focus
    updated["lexical_variation_carry_bias"] = round(lexical_variation_carry_bias, 4)
    updated["banter_move"] = surface_language_profile.banter_move
    updated["lexical_variation_mode"] = surface_language_profile.lexical_variation_mode
    updated["group_register"] = surface_language_profile.group_register
    updated["surface_language_dominant_inputs"] = list(
        surface_language_profile.dominant_inputs
    )
    updated["cues"] = _dedupe_preserve(cues)
    return updated


def _derive_qualia_planner_cues(qualia_planner_view: QualiaPlannerView) -> list[str]:
    cues: list[str] = []
    if qualia_planner_view.dominant_axis:
        cues.append("qualia_shared_input")
        cues.append(f"qualia_axis_{qualia_planner_view.dominant_axis}")
    if qualia_planner_view.degraded:
        cues.append("qualia_degraded")
    elif qualia_planner_view.trust < 0.55:
        cues.append("qualia_low_trust")
    if qualia_planner_view.felt_energy >= 0.18:
        cues.append("qualia_felt_active")
    if qualia_planner_view.body_load >= 0.08:
        cues.append("qualia_body_load")
    if qualia_planner_view.protection_bias >= 0.12:
        cues.append("qualia_protective_bias")
    return cues


def _dedupe_preserve(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _derive_task_phase(active_goals: list[str]) -> str:
    if "repair" in active_goals:
        return "repair"
    if any(goal in {"coordinate", "co_work", "shared_task"} for goal in active_goals):
        return "coordination"
    return "ongoing"


def _derive_privacy_level(*, relation_bias_strength: float, social_tension: float) -> float:
    value = 0.46 + relation_bias_strength * 0.34 - max(0.0, social_tension) * 0.14
    return max(0.0, min(1.0, value))


def _derive_norm_pressure(
    *,
    social_tension: float,
    partner_timing_hint: str,
    partner_stance_hint: str,
) -> float:
    value = max(0.0, social_tension) * 0.42
    if partner_stance_hint == "respectful":
        value += 0.28
    if partner_timing_hint == "delayed":
        value += 0.18
    return max(0.0, min(1.0, value))


def _derive_safety_margin(
    *,
    current_risks: list[str],
    arousal: float,
    social_tension: float,
) -> float:
    baseline = 0.84 - max(0.0, arousal) * 0.24 - max(0.0, social_tension) * 0.22
    if "danger" in current_risks:
        baseline -= 0.32
    return max(0.0, min(1.0, baseline))


def _derive_environmental_load(
    *,
    arousal: float,
    social_tension: float,
    current_risks: list[str],
) -> float:
    value = max(max(0.0, arousal) * 0.48, max(0.0, social_tension) * 0.54)
    if current_risks:
        value = max(value, 0.62)
    return max(0.0, min(1.0, value))
