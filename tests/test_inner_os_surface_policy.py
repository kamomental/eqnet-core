from inner_os.expression.surface_policy import (
    compile_surface_policy,
    render_surface_fallback,
)


def test_compile_surface_policy_turns_hold_contract_into_nonverbal_policy() -> None:
    policy = compile_surface_policy(
        {
            "response_channel": "hold",
            "scale": "micro",
            "question_budget": 0,
            "interpretation_budget": "low",
            "initiative": "yield",
            "distance_mode": "guarded",
            "timing_mode": "held_open",
            "closure_mode": "leave_open",
        }
    )

    assert policy.response_channel == "hold"
    assert policy.max_sentences == 0
    assert policy.question_budget == 0
    assert policy.advice_budget == 0
    assert policy.brightness_budget == 0
    assert "nonverbal_presence" in policy.allowed_acts
    assert "generate_text" in policy.prohibited_acts
    assert "ask_question" in policy.prohibited_acts
    assert policy.fallback_shape_id == "presence_hold"


def test_compile_surface_policy_keeps_speak_contract_as_surface_view() -> None:
    policy = compile_surface_policy(
        {
            "response_channel": "speak",
            "scale": "small",
            "question_budget": 0,
            "interpretation_budget": "none",
            "initiative": "receive",
            "distance_mode": "steady",
            "timing_mode": "quick_ack",
            "shape_id": "bright_bounce",
        }
    )

    assert policy.response_channel == "speak"
    assert policy.max_sentences == 1
    assert policy.question_budget == 0
    assert policy.interpretation_budget == "none"
    assert policy.advice_budget == 0
    assert policy.brightness_budget == 0
    assert "minimal_acknowledgement" in policy.allowed_acts
    assert "surface_mirror" in policy.allowed_acts
    assert "ask_question" in policy.prohibited_acts
    assert "infer_hidden_feeling" in policy.prohibited_acts
    assert "positive_spin" in policy.prohibited_acts
    assert "attribute_motive" in policy.prohibited_acts
    assert policy.fallback_shape_id == "low_inference_ack"


def test_compile_surface_policy_does_not_guard_non_bright_speak_contract() -> None:
    policy = compile_surface_policy(
        {
            "response_channel": "speak",
            "scale": "small",
            "question_budget": 0,
            "interpretation_budget": "none",
            "initiative": "receive",
            "distance_mode": "steady",
            "timing_mode": "quick_ack",
            "shape_id": "plain_ack",
        }
    )

    assert policy.max_sentences == 2
    assert "natural_surface_text" in policy.allowed_acts
    assert "positive_spin" not in policy.prohibited_acts
    assert policy.fallback_shape_id == "plain_ack_minimal"


def test_compile_surface_policy_carries_surface_style() -> None:
    policy = compile_surface_policy(
        {
            "response_channel": "speak",
            "scale": "micro",
        },
        surface_style="soft",
    )

    assert policy.surface_style == "soft"


def test_render_surface_fallback_returns_low_inference_ack_candidate() -> None:
    assert (
        render_surface_fallback(
            {
                "fallback_shape_id": "low_inference_ack",
                "fallback_variant_seed": "stable-case",
            }
        )
        in {
            "うん、そうだね。",
            "そうなのかなぁ。",
            "ああ、そういうことあるね。",
            "あるある、そういうこと。",
            "そういう時あるよね。",
        }
    )


def test_render_surface_fallback_uses_style_specific_candidates() -> None:
    assert (
        render_surface_fallback(
            {
                "fallback_shape_id": "bright_bounce_minimal",
                "surface_style": "soft",
                "fallback_variant_seed": "light-shared",
            }
        )
        in {
            "それはちょっと笑うね。",
            "あ、それは笑っちゃうね。",
            "あるある、そういうこと。",
            "そういうの、ちょっとあるよね。",
        }
    )


def test_render_surface_fallback_supports_friendly_character_style() -> None:
    assert (
        render_surface_fallback(
            {
                "fallback_shape_id": "bright_bounce_minimal",
                "surface_style": "friendly",
                "fallback_variant_seed": "friendly-light",
            }
        )
        in {
            "あ、それはちょっと笑っちゃうね。",
            "ふふ、それはあるね。",
            "あー、そういうの地味に笑うやつだ。",
        }
    )


def test_render_surface_fallback_supports_warm_playful_character_style() -> None:
    assert (
        render_surface_fallback(
            {
                "fallback_shape_id": "minimal_ack",
                "surface_style": "warm_playful",
                "fallback_variant_seed": "warm-minimal",
            }
        )
        in {
            "うんうん、ちゃんと聞いてる。",
            "そっかそっか、そういう流れね。",
            "あー、なるほどねぇ。",
        }
    )


def test_render_surface_fallback_seed_is_stable() -> None:
    policy = {
        "fallback_shape_id": "minimal_ack",
        "fallback_variant_seed": "same-seed",
    }

    assert render_surface_fallback(policy) == render_surface_fallback(policy)
