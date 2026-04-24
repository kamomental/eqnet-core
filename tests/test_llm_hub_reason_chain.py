from emot_terrain_lab.hub.llm_hub import LLMHub


def test_llm_hub_policy_prompt_keeps_reason_chain_fields() -> None:
    hub = LLMHub()

    prompt = hub._build_inner_os_policy_prompt(
        interaction_policy={"dialogue_act": "check_in"},
        actuation_plan={
            "execution_mode": "shared_progression",
            "primary_action": "riff_current_comment",
            "response_channel": "backchannel",
            "response_channel_score": 0.73,
            "reply_permission": "backchannel_or_brief",
            "wait_before_action": "brief",
            "action_queue": ["offer_backchannel_token", "keep_turn_soft"],
            "nonverbal_response_state": {
                "state": "warm_laugh_ack",
                "response_kind": "backchannel",
                "pause_mode": "micro_pause",
                "silence_mode": "porous",
                "timing_bias": "gentle_overlap",
                "token_profile": "warm_laugh",
            },
            "presence_hold_state": {
                "state": "backchannel_ready_hold",
                "silence_mode": "porous",
                "pacing_mode": "soft_return",
                "hold_room": 0.44,
                "reentry_room": 0.66,
                "backchannel_room": 0.71,
            },
        },
        reaction_contract={
            "stance": "join",
            "scale": "small",
            "initiative": "receive",
            "question_budget": 0,
            "interpretation_budget": "none",
            "response_channel": "backchannel",
            "timing_mode": "quick_ack",
            "continuity_mode": "continue",
            "distance_mode": "near",
            "closure_mode": "open_light",
            "shared_presence_mode": "inhabited_shared_space",
            "shared_presence_co_presence": 0.74,
            "shared_presence_boundary_stability": 0.68,
            "self_other_dominant_attribution": "shared",
            "self_other_unknown_likelihood": 0.12,
            "subjective_scene_anchor_frame": "shared_margin",
            "subjective_scene_shared_scene_potential": 0.7,
            "reason_tags": [
                "brief_shared_smile",
                "keep_it_small",
                "bright_bounce",
            ],
        },
        content_sequence=[{"act": "light_bounce", "text": "ok"}],
        surface_context_packet={
            "conversation_phase": "bright_continuity",
            "shared_core": {
                "anchor": "harbor",
                "already_shared": ["small laugh"],
                "not_yet_shared": [],
            },
            "response_role": {"primary": "light_bounce", "secondary": "shared_delight"},
            "constraints": {
                "no_generic_clarification": True,
                "no_advice": True,
                "max_questions": 0,
            },
            "surface_profile": {"response_length": "short", "surface_mode": "held"},
            "source_state": {
                "recent_dialogue_state": "bright_continuity",
                "discussion_thread_state": "open_thread",
                "issue_state": "light_tension",
                "turn_delta_kind": "bright_continuity",
                "appraisal_state": "active",
                "appraisal_event": "laugh_break",
                "appraisal_shared_shift": "shared_smile_window",
                "appraisal_dominant_relation_type": "same_anchor",
                "appraisal_dominant_relation_key": "harbor->promise",
                "appraisal_dominant_causal_type": "enabled_by",
                "appraisal_dominant_causal_key": "harbor->promise:enabled_by",
                "appraisal_memory_mode": "ignite",
                "appraisal_memory_anchor": "harbor",
                "appraisal_memory_resonance": 0.58,
                "joint_state": "delighted_jointness",
                "joint_shared_delight": 0.72,
                "joint_shared_tension": 0.18,
                "joint_repair_readiness": 0.51,
                "joint_common_ground": 0.67,
                "joint_attention": 0.62,
                "joint_mutual_room": 0.56,
                "joint_coupling_strength": 0.69,
                "meaning_update_state": "active",
                "meaning_update_relation": "shared_smile_window",
                "meaning_update_relation_frame": "same_anchor_link",
                "meaning_update_relation_key": "harbor->promise",
                "meaning_update_causal_frame": "same_anchor_cause",
                "meaning_update_causal_key": "harbor->promise:enabled_by",
                "meaning_update_world": "small_moment_on_known_thread",
                "meaning_update_memory": "known_thread_returns",
                "meaning_update_memory_anchor": "harbor",
                "meaning_update_memory_resonance": 0.58,
                "utterance_reason_state": "active",
                "utterance_reason_offer": "brief_shared_smile",
                "utterance_reason_preserve": "keep_it_small",
                "utterance_reason_question_policy": "none",
                "utterance_reason_relation_frame": "same_anchor_link",
                "utterance_reason_relation_key": "harbor->promise",
                "utterance_reason_causal_frame": "same_anchor_cause",
                "utterance_reason_causal_key": "harbor->promise:enabled_by",
                "utterance_reason_memory_frame": "echo_known_thread",
                "utterance_reason_memory_anchor": "harbor",
                "organism_posture": "attune",
                "organism_relation_focus": "one_to_one",
                "organism_social_mode": "near",
                "organism_attunement": 0.74,
                "organism_coherence": 0.68,
                "organism_grounding": 0.59,
                "organism_protective_tension": 0.21,
                "organism_expressive_readiness": 0.71,
                "organism_play_window": 0.64,
                "organism_relation_pull": 0.77,
                "organism_social_exposure": 0.22,
                "external_field_dominant": "shared_room",
                "external_field_social_mode": "near",
                "external_field_thread_mode": "known_thread",
                "external_field_environmental_load": 0.18,
                "external_field_social_pressure": 0.16,
                "external_field_continuity_pull": 0.62,
                "external_field_ambiguity_load": 0.12,
                "external_field_safety_envelope": 0.81,
                "external_field_novelty": 0.23,
                "terrain_dominant_basin": "shared_relief_basin",
                "terrain_dominant_flow": "ease_into_shared_smile",
                "terrain_energy": 0.42,
                "terrain_entropy": 0.24,
                "terrain_ignition_pressure": 0.37,
                "terrain_barrier_height": 0.29,
                "terrain_recovery_gradient": 0.55,
                "terrain_basin_pull": 0.61,
                "memory_dynamics_mode": "ignite",
                "memory_dominant_relation_type": "same_anchor",
                "memory_relation_generation_mode": "ignited",
                "memory_dominant_causal_type": "enabled_by",
                "memory_causal_generation_mode": "anchored",
                "memory_palace_mode": "anchored",
                "memory_monument_mode": "rising",
                "memory_ignition_mode": "active",
                "memory_reconsolidation_mode": "settle",
                "memory_recall_anchor": "harbor",
                "memory_monument_salience": 0.61,
                "memory_activation_confidence": 0.58,
                "memory_tension": 0.22,
            },
        },
    )

    assert "appraisal_shared_shift" in prompt
    assert "meaning_update_relation" in prompt
    assert "meaning_update_relation_frame" in prompt
    assert "meaning_update_causal_frame" in prompt
    assert "utterance_reason_offer" in prompt
    assert "utterance_reason_relation_frame" in prompt
    assert "utterance_reason_causal_frame" in prompt
    assert "utterance_reason_question_policy" in prompt
    assert "meaning_update_memory" in prompt
    assert "utterance_reason_memory_frame" in prompt
    assert "joint_shared_delight" in prompt
    assert "joint_common_ground" in prompt
    assert "organism_posture" in prompt
    assert "external_field_dominant" in prompt
    assert "terrain_dominant_flow" in prompt
    assert "memory_recall_anchor" in prompt
    assert "appraisal_dominant_causal_type" in prompt
    assert "memory_dominant_causal_type" in prompt
    assert "memory_causal_generation_mode" in prompt
    assert '"response_cause"' in prompt
    assert '"immediate"' in prompt
    assert '"memory_link"' in prompt
    assert '"joint_position"' in prompt
    assert '"stance"' in prompt
    assert '"reply_rule"' in prompt
    assert '"causal_frame": "same_anchor_cause"' in prompt
    assert '"causal_key": "harbor->promise:enabled_by"' in prompt
    assert '"causal_type": "enabled_by"' in prompt
    assert '"causal_generation_mode": "anchored"' in prompt
    assert '"response_channel": "backchannel"' in prompt
    assert '"reaction_contract"' in prompt
    assert '"reaction_language_guard"' in prompt
    assert '"max_sentences": 2' in prompt
    assert '出来事の詳細を聞きに行かない' in prompt
    assert '意味づけや分析を足さない' in prompt
    assert '"interpretation_budget": "none"' in prompt
    assert '"timing_mode": "quick_ack"' in prompt
    assert '"shared_presence_mode": "inhabited_shared_space"' in prompt
    assert '"self_other_dominant_attribution": "shared"' in prompt
    assert '"nonverbal_response_state"' in prompt
    assert '"presence_hold_state"' in prompt
    assert "共有された場の内側から、そのまま一緒に受ける" in prompt
    assert "外から観察して説明する語りにしない" in prompt


def test_llm_hub_policy_prompt_adds_guarded_self_view_language_guard() -> None:
    hub = LLMHub()

    prompt = hub._build_inner_os_policy_prompt(
        reaction_contract={
            "stance": "hold",
            "scale": "small",
            "initiative": "yield",
            "question_budget": 0,
            "interpretation_budget": "low",
            "response_channel": "hold",
            "timing_mode": "held_open",
            "continuity_mode": "fresh",
            "distance_mode": "guarded",
            "closure_mode": "leave_open",
            "shared_presence_mode": "guarded_boundary",
            "shared_presence_co_presence": 0.22,
            "shared_presence_boundary_stability": 0.18,
            "self_other_dominant_attribution": "unknown",
            "self_other_unknown_likelihood": 0.76,
            "subjective_scene_anchor_frame": "ambient_margin",
            "subjective_scene_shared_scene_potential": 0.18,
            "reason_tags": ["guarded_boundary", "do_not_overclaim"],
        }
    )

    assert '"shared_presence_mode": "guarded_boundary"' in prompt
    assert '"self_other_dominant_attribution": "unknown"' in prompt
    assert "境界を保ったまま、踏み込みすぎずに返す" in prompt
    assert "親密さを勝手に前提化しない" in prompt
