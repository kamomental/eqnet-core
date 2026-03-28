# -*- coding: utf-8 -*-

from inner_os.expression.surface_context_packet import build_surface_context_packet


def test_build_surface_context_packet_keeps_reopening_anchor_and_constraints() -> None:
    packet = build_surface_context_packet(
        recent_dialogue_state={
            "state": "reopening_thread",
            "recent_anchor": "前に少し引っかかっていた話",
        },
        discussion_thread_state={
            "state": "revisit_issue",
            "topic_anchor": "あの約束",
        },
        issue_state={"state": "pausing_issue"},
        turn_delta={
            "kind": "reopening_thread",
            "preferred_act": "reopen_from_anchor",
            "anchor_hint": "あの約束",
        },
        interaction_constraints={
            "avoid_obvious_advice": True,
            "keep_thread_visible": True,
            "prefer_return_point": True,
        },
        boundary_transform={"surface_mode": "soft_close"},
        residual_reflection={"focus": "言えていないこと", "reasons": ["withheld_need"]},
        surface_profile={
            "response_length": "short",
            "cultural_register": "casual_shared",
            "group_register": "one_to_one",
            "sentence_temperature": "gentle",
        },
        contact_reflection_state={"reflection_style": "reflect_only"},
        green_kernel_composition={
            "field": {
                "guardedness": 0.72,
                "reopening_pull": 0.64,
                "affective_charge": 0.58,
            }
        },
        dialogue_context={"user_text": "その続きなら、いま話せるところからで。"},
    ).to_dict()

    assert packet["conversation_phase"] == "reopening_thread"
    assert packet["shared_core"]["anchor"] == "あの約束"
    assert "あの約束" in packet["shared_core"]["already_shared"]
    assert "言えていないこと" in packet["shared_core"]["not_yet_shared"]
    assert packet["response_role"]["primary"] == "reopen_from_anchor"
    assert packet["response_role"]["secondary"] == "reflect_only"
    assert packet["constraints"]["no_generic_clarification"] is True
    assert packet["constraints"]["no_advice"] is True
    assert packet["constraints"]["max_questions"] == 0
    assert packet["constraints"]["keep_thread_visible"] is True
    assert packet["surface_profile"]["cultural_register"] == "casual_shared"
    assert packet["source_state"]["green_guardedness"] == 0.72
