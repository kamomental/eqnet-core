from __future__ import annotations

from inner_os.distillation_record import InnerOSDistillationRecordBuilder
from inner_os.schemas import (
    INNER_OS_DISTILLATION_RECORD_SCHEMA,
    distillation_record_contract,
)


def test_distillation_record_builder_redacts_text_by_default() -> None:
    builder = InnerOSDistillationRecordBuilder()
    record = builder.build(
        turn_id="hub-1",
        session_id="session-1",
        timestamp_ms=1234,
        user_text="Can you stay with what is visible first?",
        context_text="known context",
        response_text="I will stay with what is visible first.",
        response_meta={
            "model": "qwen-3.5-instruct",
            "model_source": "live_list",
            "trace_id": "hub-1",
            "latency_ms": 12.5,
            "confidence": 0.74,
        },
        interaction_policy_packet={
            "protection_mode_decision": {"mode": "repair", "winner_margin": 0.32},
            "memory_write_class_bias": {"selected_class": "repair_trace", "winner_margin": 0.21},
            "body_recovery_guard": {"state": "guarded", "score": 0.44},
            "body_homeostasis_state": {"state": "recovering", "score": 0.48, "winner_margin": 0.14},
            "initiative_readiness": {"state": "tentative", "score": 0.38},
            "commitment_state": {"state": "settle", "target": "repair", "accepted_cost": 0.28},
            "relational_style_memory_state": {
                "state": "warm_companion",
                "playful_ceiling": 0.42,
                "advice_tolerance": 0.56,
                "lexical_variation_bias": 0.34,
                "banter_style": "gentle_tease",
            },
            "cultural_conversation_state": {"state": "public_courteous", "directness_ceiling": 0.28, "joke_ratio_ceiling": 0.18},
            "expressive_style_state": {"state": "warm_companion", "lightness_room": 0.42, "continuity_weight": 0.58},
            "lightness_budget_state": {"state": "warm_only", "banter_room": 0.3, "suppression": 0.18},
            "relational_continuity_state": {"state": "reopening", "score": 0.52, "winner_margin": 0.12},
            "initiative_followup_bias": {"state": "reopen_softly", "score": 0.31},
            "overnight_bias_roles": {"association": "repeated_links"},
            "reaction_vs_overnight_bias": {"overnight": {"association_focus": "repeated_links"}},
            "expressive_style_history_focus": "warm_companion",
            "expressive_style_history_bias": 0.08,
            "banter_style_focus": "gentle_tease",
            "lexical_variation_carry_bias": 0.11,
        },
        persona_meta_inner_os={
            "workspace_decision": {"workspace_mode": "meaning", "winner_margin": 0.41},
            "surface_policy_level": "careful",
            "surface_policy_intent": "repair",
            "route": "watch",
            "talk_mode": "watch",
        },
        include_text=False,
    ).to_dict()

    assert record["schema"] == INNER_OS_DISTILLATION_RECORD_SCHEMA
    assert record["model"]["source"] == "live_list"
    assert record["input_fingerprint"]["user_text_sha256"]
    assert record["output_fingerprint"]["response_text_sha256"]
    assert record["text_payload"] == {}
    assert record["decision_snapshot"]["protection_mode"]["mode"] == "repair"
    assert record["decision_snapshot"]["body_homeostasis_state"]["state"] == "recovering"
    assert record["decision_snapshot"]["relational_style_memory_state"]["state"] == "warm_companion"
    assert record["decision_snapshot"]["relational_style_memory_state"]["banter_style"] == "gentle_tease"
    assert record["decision_snapshot"]["cultural_conversation_state"]["state"] == "public_courteous"
    assert record["decision_snapshot"]["expressive_style_state"]["state"] == "warm_companion"
    assert record["decision_snapshot"]["lightness_budget_state"]["state"] == "warm_only"
    assert record["decision_snapshot"]["relational_continuity_state"]["state"] == "reopening"
    assert record["carry_snapshot"]["expressive_style_history"]["focus"] == "warm_companion"
    assert record["carry_snapshot"]["expressive_style_history"]["banter_style_focus"] == "gentle_tease"


def test_distillation_record_contract_exposes_required_fields() -> None:
    contract = distillation_record_contract()

    assert contract["schema"] == INNER_OS_DISTILLATION_RECORD_SCHEMA
    assert "model" in contract["required_fields"]
    assert "decision_snapshot" in contract["required_fields"]
    assert "text_payload" in contract["optional_fields"]
