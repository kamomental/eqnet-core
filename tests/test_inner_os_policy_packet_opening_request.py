from inner_os.policy_packet import (
    InteractionPolicyPacketContract,
    derive_interaction_policy_packet,
)


class _PolicyRegulation:
    repair_window_open = False
    strained_pause = 0.06
    future_loop_pull = 0.12
    fantasy_loop_pull = 0.0
    distance_expectation = "holding_space"


def test_policy_packet_keeps_small_opening_support_for_explicit_clarify_request() -> None:
    packet = derive_interaction_policy_packet(
        dialogue_act="clarify",
        observed_text="今のしんどさを無理に整理せず、何が引っかかっているかだけ一緒に見てほしいです。どう切り出せばよさそうですか。",
        locale="ja-JP",
        current_focus="social",
        current_risks=[],
        reportable_facts=["social"],
        relation_bias_strength=0.0,
        related_person_ids=[],
        partner_address_hint="",
        partner_timing_hint="",
        partner_stance_hint="",
        orchestration={
            "orchestration_mode": "attune",
            "dominant_driver": "shared_attention",
            "contact_readiness": 0.48,
            "coherence_score": 0.61,
            "human_presence_signal": 0.44,
            "distance_strategy": "holding_space",
            "repair_bias": False,
        },
        surface_profile={
            "opening_pace_windowed": "ready",
            "return_gaze_expectation": "soft_return",
        },
        live_regulation=_PolicyRegulation(),
        interaction_option_candidates=[
            {
                "family_id": "attune",
                "option_id": "attune:test",
            }
        ],
        object_operations={
            "primary_operation_id": "operation:0",
            "operations": [
                {
                    "operation_id": "operation:0",
                    "operation_kind": "acknowledge",
                    "target_label": "social",
                },
                {
                    "operation_id": "operation:1",
                    "operation_kind": "keep_return_point",
                    "target_label": "social",
                },
            ],
            "question_budget": 1,
            "question_pressure": 0.5,
            "defer_dominance": 0.7,
        },
    )

    assert packet["opening_request_hint"] is True
    assert packet["dialogue_act"] == "clarify"
    assert packet["response_strategy"] == "contain_then_stabilize"
    assert packet["opening_move"] == "acknowledge_without_probe"
    assert packet["question_budget"] == 0
    assert packet["effective_question_budget"] == 0


def test_policy_packet_is_mutable_mapping_contract() -> None:
    packet = derive_interaction_policy_packet(
        dialogue_act="report",
        observed_text="",
        locale="ja-JP",
        current_focus="ambient",
        current_risks=[],
        reportable_facts=["ambient"],
        relation_bias_strength=0.0,
        related_person_ids=[],
        partner_address_hint="",
        partner_timing_hint="",
        partner_stance_hint="",
        orchestration={
            "orchestration_mode": "attune",
            "dominant_driver": "shared_attention",
            "contact_readiness": 0.42,
            "coherence_score": 0.48,
            "human_presence_signal": 0.36,
            "distance_strategy": "holding_space",
            "repair_bias": False,
        },
        surface_profile={
            "opening_pace_windowed": "ready",
            "return_gaze_expectation": "soft_return",
        },
        live_regulation=_PolicyRegulation(),
    )

    assert isinstance(packet, InteractionPolicyPacketContract)
    packet["posthoc_marker"] = "ok"
    assert dict(packet)["posthoc_marker"] == "ok"
    assert packet["dialogue_act"] == "report"
