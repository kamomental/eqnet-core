from inner_os.lightness_budget_state import derive_lightness_budget_state
from inner_os.relational_style_memory import derive_relational_style_memory_state


def test_relational_style_memory_prefers_playful_room_for_familiar_partner() -> None:
    state = derive_relational_style_memory_state(
        self_state={
            "related_person_id": "user",
            "attachment": 0.72,
            "familiarity": 0.76,
            "trust_memory": 0.7,
            "continuity_score": 0.66,
            "recent_strain": 0.08,
            "exploration_bias": 0.74,
            "caution_bias": 0.18,
            "social_grounding": 0.62,
            "partner_address_hint": "companion",
            "partner_timing_hint": "open",
            "partner_stance_hint": "familiar",
            "partner_social_interpretation": "familiar:future_open",
            "person_registry_snapshot": {
                "persons": {
                    "user": {
                        "adaptive_traits": {
                            "style_warmth_memory": 0.68,
                            "playful_ceiling": 0.72,
                            "advice_tolerance": 0.58,
                            "lexical_familiarity": 0.64,
                            "lexical_variation_bias": 0.42,
                        }
                    }
                }
            },
        },
        relation_bias_strength=0.62,
        related_person_ids=["user"],
        relation_competition_state={"state": "dominant_thread", "competition_level": 0.08, "dominant_person_id": "user", "total_people": 1},
        social_topology_state={"state": "one_to_one", "score": 0.14},
        body_recovery_guard={"state": "open", "score": 0.08},
        body_homeostasis_state={"state": "steady", "score": 0.14},
    ).to_dict()

    assert state["dominant_person_id"] == "user"
    assert state["playful_ceiling"] >= 0.68
    assert state["banter_room"] >= 0.45
    assert state["lexical_variation_bias"] >= 0.35
    assert state["banter_style"] in {"gentle_tease", "compact_wit", "warm_refrain", "respectful_light", "grounded_companion"}
    assert state["state"] in {"light_playful", "warm_companion"}


def test_lightness_budget_suppresses_play_under_public_guarded_load() -> None:
    budget = derive_lightness_budget_state(
        body_recovery_guard={"state": "guarded", "score": 0.52},
        body_homeostasis_state={"state": "recovering", "score": 0.44},
        homeostasis_budget_state={"state": "depleted", "score": 0.46},
        protection_mode={"mode": "contain", "strength": 0.58},
        attention_regulation_state={"state": "reflex_guard", "score": 0.62},
        grice_guard_state={"state": "hold_obvious_advice", "score": 0.72},
        social_topology_state={"state": "public_visible", "visibility_pressure": 0.76},
        relation_competition_state={"state": "competing_threads", "competition_level": 0.48},
        expressive_style_state={"state": "light_playful", "lightness_room": 0.54},
        relational_style_memory_state={
            "state": "light_playful",
            "playful_ceiling": 0.7,
            "advice_tolerance": 0.46,
            "lexical_variation_bias": 0.34,
            "banter_room": 0.5,
            "banter_style": "respectful_light",
        },
        cultural_conversation_state={
            "state": "public_courteous",
            "joke_ratio_ceiling": 0.22,
            "politeness_pressure": 0.62,
            "group_attunement": 0.28,
        },
    ).to_dict()

    assert budget["state"] in {"grounded_only", "suppress_play"}
    assert budget["suppression"] >= 0.45
    assert budget["banter_style"] == "respectful_light"
    assert budget["lexical_variation_bias"] == 0.34
