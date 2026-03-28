from types import SimpleNamespace

from emot_terrain_lab.hub.runtime import EmotionalHubRuntime


def test_apply_inner_os_surface_profile_exposes_boundary_and_residual_controls() -> None:
    runtime = EmotionalHubRuntime()
    response = SimpleNamespace(
        text="temporary raw text",
        controls_used={},
    )

    shaped = runtime._apply_inner_os_surface_profile(
        response,
        {
            "interaction_policy_packet": {
                "dialogue_act": "check_in",
                "response_strategy": "respectful_wait",
            },
            "planned_content_sequence": [
                {"act": "respect_boundary", "text": "いまは、ここを無理に押し進めなくて大丈夫です。"},
                {"act": "leave_return_point", "text": "戻るなら、あとで戻れる形で置いておけます。"},
            ],
            "boundary_transform": {
                "gate_mode": "narrow",
                "transformation_mode": "soften",
                "softened_acts": ["offer_small_opening_line"],
                "withheld_acts": [],
            },
            "residual_reflection": {
                "mode": "softened",
                "focus": "offer_small_opening_line",
                "strength": 0.42,
                "reason_tokens": ["softened_candidate"],
            },
        },
    )

    assert shaped is not None
    assert shaped.controls_used["inner_os_boundary_transform"]["gate_mode"] == "narrow"
    assert shaped.controls_used["inner_os_residual_reflection"]["mode"] == "softened"

