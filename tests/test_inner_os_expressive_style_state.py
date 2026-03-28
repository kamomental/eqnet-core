from inner_os.expressive_style_state import derive_expressive_style_state
from inner_os.expression.response_planner import _apply_interaction_policy_surface_bias


def test_expressive_style_state_prefers_light_playful_when_safe_and_open() -> None:
    expressive = derive_expressive_style_state(
        self_state={
            "stress": 0.12,
            "recovery_need": 0.08,
            "recent_strain": 0.06,
            "continuity_score": 0.74,
            "social_grounding": 0.68,
            "relation_seed_strength": 0.44,
            "long_term_theme_strength": 0.38,
            "conscious_residue_strength": 0.22,
            "expressive_style_history_focus": "warm_companion",
            "expressive_style_history_bias": 0.09,
            "banter_style_focus": "gentle_tease",
            "lexical_variation_carry_bias": 0.11,
        },
        temperament_estimate={
            "bond_drive": 0.38,
            "curiosity_drive": 0.9,
            "recovery_discipline": 0.28,
            "protect_floor": 0.2,
            "initiative_persistence": 0.82,
            "leader_tendency": 0.26,
            "hero_tendency": 0.78,
            "forward_trace": 0.58,
            "bond_trace": 0.26,
        },
        body_recovery_guard={"state": "open", "score": 0.08},
        body_homeostasis_state={"state": "steady", "score": 0.18},
        homeostasis_budget_state={"state": "steady", "score": 0.16},
        initiative_readiness={"state": "ready", "score": 0.74},
        commitment_state={"target": "step_forward", "score": 0.56},
        relational_continuity_state={"state": "co_regulating", "score": 0.54},
        relational_style_memory_state={
            "state": "light_playful",
            "score": 0.62,
            "warmth_bias": 0.42,
            "playful_ceiling": 0.72,
            "advice_tolerance": 0.56,
            "lexical_familiarity": 0.48,
            "lexical_variation_bias": 0.4,
            "banter_room": 0.58,
            "banter_style": "gentle_tease",
        },
        cultural_conversation_state={
            "state": "casual_shared",
            "score": 0.58,
            "directness_ceiling": 0.64,
            "joke_ratio_ceiling": 0.62,
            "politeness_pressure": 0.18,
            "group_attunement": 0.22,
        },
        social_topology_state={"state": "one_to_one", "score": 0.14},
        attention_regulation_state={"state": "selective_hold", "score": 0.58},
        grice_guard_state={"state": "advise_openly", "score": 0.1},
        protection_mode={"mode": "monitor", "strength": 0.18},
        contact_readiness=0.72,
        coherence_score=0.7,
        human_presence_signal=0.74,
    ).to_dict()

    assert expressive["state"] == "light_playful"
    assert expressive["lightness_room"] >= 0.45
    assert expressive["continuity_weight"] >= 0.45
    assert expressive["winner_margin"] >= 0.0
    assert any(
        item in expressive["dominant_inputs"]
        for item in ("relational_lexical_variation", "overnight_expressive_style_history", "overnight_banter_style_gentle_tease")
    )


def test_expressive_style_state_prefers_reverent_measured_when_formal_and_guarded() -> None:
    expressive = derive_expressive_style_state(
        self_state={
            "stress": 0.44,
            "recovery_need": 0.46,
            "recent_strain": 0.32,
            "continuity_score": 0.52,
            "social_grounding": 0.56,
            "relation_seed_strength": 0.4,
            "long_term_theme_strength": 0.28,
            "conscious_residue_strength": 0.26,
        },
        temperament_estimate={
            "bond_drive": 0.48,
            "curiosity_drive": 0.34,
            "recovery_discipline": 0.72,
            "protect_floor": 0.68,
            "initiative_persistence": 0.28,
            "leader_tendency": 0.36,
            "hero_tendency": 0.12,
            "forward_trace": 0.12,
            "bond_trace": 0.22,
        },
        body_recovery_guard={"state": "guarded", "score": 0.58},
        body_homeostasis_state={"state": "recovering", "score": 0.42},
        homeostasis_budget_state={"state": "depleted", "score": 0.54},
        initiative_readiness={"state": "hold", "score": 0.12},
        commitment_state={"target": "stabilize", "score": 0.44},
        relational_continuity_state={"state": "holding_thread", "score": 0.48},
        relational_style_memory_state={
            "state": "reverent_measured",
            "score": 0.58,
            "warmth_bias": 0.22,
            "playful_ceiling": 0.12,
            "advice_tolerance": 0.24,
            "lexical_familiarity": 0.18,
            "lexical_variation_bias": 0.08,
            "banter_room": 0.12,
            "banter_style": "soft_formal",
        },
        cultural_conversation_state={
            "state": "public_courteous",
            "score": 0.66,
            "directness_ceiling": 0.28,
            "joke_ratio_ceiling": 0.14,
            "politeness_pressure": 0.72,
            "group_attunement": 0.34,
        },
        social_topology_state={"state": "public_visible", "score": 0.72, "visibility_pressure": 0.78},
        attention_regulation_state={"state": "reflex_guard", "score": 0.62},
        grice_guard_state={"state": "hold_obvious_advice", "score": 0.78},
        protection_mode={"mode": "contain", "strength": 0.62},
        contact_readiness=0.34,
        coherence_score=0.42,
        human_presence_signal=0.46,
    ).to_dict()

    assert expressive["state"] == "reverent_measured"
    assert expressive["lightness_room"] <= 0.45
    assert "reverent_topology" in expressive["dominant_inputs"]


def test_surface_bias_applies_expressive_style_voice_texture() -> None:
    updated = _apply_interaction_policy_surface_bias(
        {
            "opening_delay": "brief",
            "response_length": "balanced",
            "sentence_temperature": "neutral",
            "pause_insertion": "none",
            "certainty_style": "direct",
            "cues": [],
        },
        {
            "body_recovery_guard": {"state": "open", "score": 0.08},
            "body_homeostasis_state": {"state": "steady", "score": 0.18},
            "homeostasis_budget_state": {"state": "steady", "score": 0.16},
            "initiative_readiness": {"state": "ready", "score": 0.62},
            "commitment_state": {"state": "settle", "target": "step_forward", "score": 0.48},
            "identity_arc_kind": "repairing_bond",
            "identity_arc_phase": "integrating",
            "identity_arc_summary": "repair is gathering around a relationship thread",
            "identity_arc_open_tension": "timing_sensitive_reentry",
            "identity_arc_stability": 0.58,
            "attention_regulation_state": {"state": "selective_hold", "score": 0.52},
            "grice_guard_state": {"state": "advise_openly", "score": 0.12},
            "relational_style_memory_state": {
                "state": "light_playful",
                "score": 0.62,
                "playful_ceiling": 0.7,
                "advice_tolerance": 0.54,
                "lexical_variation_bias": 0.36,
                "banter_room": 0.58,
                "banter_style": "gentle_tease",
            },
            "cultural_conversation_state": {
                "state": "casual_shared",
                "score": 0.58,
                "directness_ceiling": 0.64,
                "joke_ratio_ceiling": 0.62,
                "politeness_pressure": 0.18,
                "group_attunement": 0.22,
            },
            "expressive_style_state": {
                "state": "light_playful",
                "score": 0.72,
                "lightness_room": 0.66,
                "continuity_weight": 0.54,
            },
            "lightness_budget_state": {
                "state": "open_play",
                "score": 0.68,
                "banter_room": 0.62,
                "playful_ceiling": 0.7,
                "suppression": 0.12,
            },
            "relational_continuity_state": {"state": "co_regulating", "score": 0.58},
            "social_topology_state": {"state": "one_to_one", "score": 0.12},
            "protection_mode": {"mode": "monitor", "strength": 0.18},
            "expressive_style_history_focus": "warm_companion",
            "expressive_style_history_bias": 0.08,
            "banter_style_focus": "gentle_tease",
            "lexical_variation_carry_bias": 0.1,
        },
    )

    assert updated["voice_texture"] == "light_playful"
    assert updated["relational_voice_texture"] == "light_playful"
    assert updated["lightness_room"] == 0.66
    assert updated["continuity_weight"] == 0.54
    assert updated["cultural_register"] == "casual_shared"
    assert updated["lightness_budget_state"] == "open_play"
    assert updated["sentence_temperature"] == "warm"
    assert updated["relational_banter_style"] == "gentle_tease"
    assert updated["relational_lexical_variation_bias"] == 0.36
    assert updated["identity_arc_kind"] == "repairing_bond"
    assert updated["identity_arc_phase"] == "integrating"
    assert updated["identity_arc_open_tension"] == "timing_sensitive_reentry"
    assert updated["expressive_style_history_focus"] == "warm_companion"
    assert updated["banter_style_focus"] == "gentle_tease"
    assert updated["lexical_variation_carry_bias"] == 0.1
    assert updated["banter_move"] == "warm_refrain"
    assert updated["lexical_variation_mode"] == "warm_varied"
    assert updated["group_register"] == "one_to_one"
    assert "surface_identity_repairing_bond" in updated["cues"]
    assert "surface_identity_timing_sensitive_reentry" in updated["cues"]
    assert "surface_expressive_light_playful" in updated["cues"]
    assert "surface_language_banter_warm_refrain" in updated["cues"]
