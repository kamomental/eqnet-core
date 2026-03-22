from inner_os.expression.surface_language_profile import (
    derive_surface_language_profile,
    shape_surface_language_text,
)


def test_surface_language_profile_prefers_compact_wit_in_safe_one_to_one() -> None:
    profile = derive_surface_language_profile(
        recovery_state="open",
        protection_mode_name="monitor",
        grice_state="advise_openly",
        expressive_style_name="light_playful",
        expressive_style_history_focus="warm_companion",
        relational_continuity_name="co_regulating",
        relational_banter_style="compact_wit",
        relational_lexical_variation_bias=0.48,
        relational_banter_room=0.54,
        lightness_budget_name="open_play",
        lightness_banter_room=0.58,
        lightness_playful_ceiling=0.64,
        lightness_suppression=0.12,
        social_topology_name="one_to_one",
        cultural_state_name="casual_shared",
        cultural_joke_ratio_ceiling=0.62,
        lexical_variation_carry_bias=0.12,
    )

    assert profile.banter_move == "compact_wit"
    assert profile.lexical_variation_mode == "compact_varied"
    assert profile.group_register == "one_to_one"


def test_surface_language_profile_prefers_thread_soften_in_group() -> None:
    profile = derive_surface_language_profile(
        recovery_state="open",
        protection_mode_name="monitor",
        grice_state="acknowledge_then_extend",
        expressive_style_name="warm_companion",
        expressive_style_history_focus="warm_companion",
        relational_continuity_name="holding_thread",
        relational_banter_style="warm_refrain",
        relational_lexical_variation_bias=0.3,
        relational_banter_room=0.3,
        lightness_budget_name="warm_only",
        lightness_banter_room=0.34,
        lightness_playful_ceiling=0.28,
        lightness_suppression=0.18,
        social_topology_name="threaded_group",
        cultural_state_name="group_attuned",
        cultural_joke_ratio_ceiling=0.26,
        lexical_variation_carry_bias=0.08,
    )

    assert profile.banter_move == "thread_soften"
    assert profile.lexical_variation_mode == "group_attuned"
    assert profile.group_register == "threaded_group"


def test_shape_surface_language_text_applies_compact_and_group_moves() -> None:
    compact = shape_surface_language_text(
        "I am trying to stay with what is grounded.",
        surface_profile={
            "banter_move": "compact_wit",
            "lexical_variation_mode": "compact_varied",
            "group_register": "one_to_one",
        },
    )
    grouped = shape_surface_language_text(
        "I can stay with what is visible first.",
        surface_profile={
            "banter_move": "thread_soften",
            "lexical_variation_mode": "group_attuned",
            "group_register": "threaded_group",
        },
    )

    assert compact.startswith("Short version:")
    assert "I'm trying" in compact
    assert grouped.startswith("For this thread,")
