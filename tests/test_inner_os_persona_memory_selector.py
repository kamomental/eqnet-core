from inner_os.persona_memory_fragment import build_persona_memory_fragments
from inner_os.persona_memory_selector import derive_persona_memory_selection


def test_persona_memory_selector_prefers_relation_and_boundary_in_public_reentry() -> None:
    fragments = build_persona_memory_fragments(
        self_state={
            "identity_arc_kind": "repairing_bond",
            "identity_arc_summary": "repair is holding around the harbor thread",
            "identity_arc_phase": "integrating",
            "identity_arc_stability": 0.61,
            "memory_anchor": "harbor slope",
            "related_person_id": "person:harbor",
            "group_thread_focus": "threaded_group:person:harbor|person:friend",
            "long_term_theme_kind": "meaning",
            "long_term_theme_focus": "quiet harbor promise",
            "long_term_theme_summary": "quiet harbor promise",
            "long_term_theme_strength": 0.52,
            "relation_seed_summary": "shared harbor repair line",
            "relation_seed_strength": 0.66,
            "partner_timing_hint": "delayed",
            "partner_stance_hint": "respectful",
            "partner_social_interpretation": "respectful:delayed",
        },
        relation_bias_strength=0.72,
        related_person_ids=["person:harbor"],
        social_topology_state={"state": "public_visible", "visibility_pressure": 0.64},
        relational_style_memory_state={
            "state": "warm_attuned",
            "playful_ceiling": 0.22,
            "advice_tolerance": 0.28,
            "banter_style": "thread_soften",
        },
        cultural_conversation_state={
            "state": "public_courteous",
            "directness_ceiling": 0.34,
        },
        protection_mode={"mode": "contain", "strength": 0.58},
        grice_guard_state={"state": "hold_obvious_advice", "knownness_pressure": 0.62},
    )
    selection = derive_persona_memory_selection(
        fragments=fragments,
        current_focus="harbor repair thread",
        reportable_facts=["repair is possible but public pressure is high"],
        current_risks=["public_attention"],
        relation_bias_strength=0.72,
        agenda_window_state={"state": "next_private_window"},
        social_topology_state={"state": "public_visible"},
        grice_guard_state={"state": "hold_obvious_advice", "knownness_pressure": 0.62},
    )

    assert selection.dominant_fragment_id in {"relation_seed", "boundary_memory"}
    assert "boundary_memory" in selection.selected_fragment_ids
    assert "relation_seed" in selection.selected_fragment_ids
    assert selection.scores["boundary_memory"] > 0.24
    assert selection.scores["relation_seed"] > 0.24


def test_persona_memory_selector_prefers_theme_in_same_culture_reentry() -> None:
    fragments = build_persona_memory_fragments(
        self_state={
            "long_term_theme_kind": "place",
            "long_term_theme_focus": "quiet harbor routine",
            "long_term_theme_summary": "quiet harbor routine",
            "long_term_theme_anchor": "harbor slope",
            "long_term_theme_strength": 0.64,
        },
        relation_bias_strength=0.18,
        related_person_ids=[],
        social_topology_state={"state": "ambient"},
        relational_style_memory_state={"state": "grounded_gentle"},
        cultural_conversation_state={"state": "group_attuned", "directness_ceiling": 0.46},
        protection_mode={"mode": "monitor", "strength": 0.18},
        grice_guard_state={"state": "acknowledge_then_extend", "knownness_pressure": 0.18},
    )
    selection = derive_persona_memory_selection(
        fragments=fragments,
        current_focus="harbor routine",
        reportable_facts=["the same cultural meeting may reopen this topic"],
        current_risks=[],
        relation_bias_strength=0.18,
        agenda_window_state={"state": "next_same_culture_window"},
        social_topology_state={"state": "threaded_group"},
        grice_guard_state={"state": "acknowledge_then_extend", "knownness_pressure": 0.18},
    )

    assert selection.dominant_fragment_id == "long_term_theme"
    assert selection.selected_fragment_ids[0] == "long_term_theme"
