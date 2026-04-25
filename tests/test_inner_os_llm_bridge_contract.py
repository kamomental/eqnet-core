# -*- coding: utf-8 -*-

from inner_os.expression.llm_bridge_contract import review_llm_bridge_text


def _small_shared_smile_packet() -> dict[str, object]:
    return {
        "conversation_phase": "bright_continuity",
        "constraints": {"max_questions": 0},
        "source_state": {
            "utterance_reason_offer": "brief_shared_smile",
            "utterance_reason_preserve": "keep_it_small",
            "utterance_reason_question_policy": "none",
            "shared_moment_kind": "laugh",
        },
    }


def test_review_llm_bridge_text_replaces_small_shared_smile_question_with_fallback() -> None:
    review = review_llm_bridge_text(
        raw_text="そうだったんですね。少し笑えたんですね。その後のお気持ちはどうでしたか？",
        surface_context_packet=_small_shared_smile_packet(),
        fallback_text="ふふっ、それ、ちょっと笑えるやつだね。 そういうのあると、ちょっと楽になるよね。",
    )

    assert review.ok is False
    assert review.sanitized_text == "ふふっ、それ、ちょっと笑えるやつだね。 そういうのあると、ちょっと楽になるよね。"
    assert "question_block_violation" in review.violation_codes()
    assert "assistant_attractor_violation" in review.violation_codes()


def test_review_llm_bridge_text_keeps_small_shared_smile_chat_reaction() -> None:
    review = review_llm_bridge_text(
        raw_text="ふふっ、それ、ちょっと笑えるやつだね。 そういうのあると、ちょっと楽になるよね。",
        surface_context_packet=_small_shared_smile_packet(),
    )

    assert review.ok is True
    assert review.sanitized_text == "ふふっ、それ、ちょっと笑えるやつだね。 そういうのあると、ちょっと楽になるよね。"
    assert review.violation_codes() == []


def test_review_llm_bridge_text_blocks_interpretive_probe_style_for_small_shared_smile() -> None:
    review = review_llm_bridge_text(
        raw_text=(
            "おっしゃる通りですね。その後の出来事が少し和み、笑えたという感覚は、重たいものから離れる一歩かもしれませんね。"
            "\n\n"
            "ただ、その笑えることが何だったのか、もしお話ししそうなことがあれば、気兼ねなく触れても良いかもしれません。"
            "今の気分は、それによってどう変化しましたか？"
            "\n\n"
            "※推定信頼度: 0.78（中） / 不確実要因: 低"
        ),
        surface_context_packet=_small_shared_smile_packet(),
        fallback_text="ふふっ、それ、ちょっと笑えるやつだね。 そういうのあると、ちょっと楽になるよね。",
    )

    assert review.ok is False
    assert review.sanitized_text == "ふふっ、それ、ちょっと笑えるやつだね。 そういうのあると、ちょっと楽になるよね。"
    assert "interpretive_bright_violation" in review.violation_codes()
    assert "uncertainty_meta_violation" in review.violation_codes()


def test_review_llm_bridge_text_blocks_uncertainty_meta_even_without_small_shared_moment() -> None:
    review = review_llm_bridge_text(
        raw_text="少し笑えてよかったです。※推定信頼度: 0.78（中） / 不確実要因: 低",
        surface_context_packet={
            "conversation_phase": "clarify",
            "constraints": {"max_questions": 1},
            "source_state": {},
        },
        fallback_text="少し笑えてよかったです。",
    )

    assert review.ok is False
    assert review.sanitized_text == "少し笑えてよかったです。"
    assert "uncertainty_meta_violation" in review.violation_codes()


def test_review_llm_bridge_text_blocks_live_like_supportive_prompting() -> None:
    review = review_llm_bridge_text(
        raw_text=(
            "さて、さっきの話の続きですね。"
            "その後に少し笑えた出来事があったとのこと。"
            "今、どのような感覚を抱いているか、ゆっくりと見守ってください。"
            "※推定信頼度: 0.78（中） / 不確実要因: 低"
        ),
        surface_context_packet=_small_shared_smile_packet(),
        fallback_text="ふふっ、それ、ちょっと笑えるやつだね。 そういうのあると、ちょっと楽になるよね。",
    )

    assert review.ok is False
    assert review.sanitized_text == "ふふっ、それ、ちょっと笑えるやつだね。 そういうのあると、ちょっと楽になるよね。"
    assert "elicitation_violation" in review.violation_codes()
    assert "assistant_attractor_violation" in review.violation_codes()
    assert "uncertainty_meta_violation" in review.violation_codes()


def test_review_llm_bridge_text_blocks_interpretive_sentence_without_question_mark() -> None:
    review = review_llm_bridge_text(
        raw_text=(
            "そうなんだね。少し笑えた瞬間が、その後のしんどさがやわらぐきっかけになったんだろうな。"
            "ただの出来事というよりも、心が少し緩んだ証拠のように思える。"
        ),
        surface_context_packet=_small_shared_smile_packet(),
        fallback_text="ふふっ、それ、ちょっと笑えるやつだね。 そういうのあると、ちょっと楽になるよね。",
    )

    assert review.ok is False
    assert review.sanitized_text == "ふふっ、それ、ちょっと笑えるやつだね。 そういうのあると、ちょっと楽になるよね。"
    assert "interpretive_bright_violation" in review.violation_codes()


def test_review_llm_bridge_text_treats_bright_bounce_discourse_shape_as_small_shared_moment() -> None:
    review = review_llm_bridge_text(
        raw_text=(
            "お疲れ様でした。"
            "その後の出来事が少し笑いになっていったということは、肩が少し軽くなったのかもしれませんね。"
            "今はその感覚をそのまま受け止めていて大丈夫です。"
        ),
        surface_context_packet={
            "conversation_phase": "continuing_thread",
            "constraints": {"max_questions": 0},
            "surface_profile": {"discourse_shape_id": "bright_bounce"},
            "source_state": {
                "discourse_shape_id": "bright_bounce",
                "utterance_reason_question_policy": "none",
            },
        },
        fallback_text="それ、ちょっと笑えるやつだね。 そういうのあると、ちょっと楽になるよね。",
    )

    assert review.ok is False
    assert review.sanitized_text == "それ、ちょっと笑えるやつだね。 そういうのあると、ちょっと楽になるよね。"
    assert "interpretive_bright_violation" in review.violation_codes()


def test_review_llm_bridge_text_blocks_live_like_followup_question_sequence() -> None:
    review = review_llm_bridge_text(
        raw_text=(
            "おっしゃる通りですね。"
            "昨日の続きですね。"
            "その後は少し笑えることがあったようです。"
            "そんな小さな瞬間も、心の中に溜まっていくのかもしれませんね。"
            "今はどうお過ごしでしたか？"
            "\n\n※推定信頼度: 0.78（中） / 不確実要因: 低"
        ),
        surface_context_packet=_small_shared_smile_packet(),
        fallback_text="ふふっ、それ、ちょっと笑えるやつだね。 そういうのあると、ちょっと楽になるよね。",
    )

    assert review.ok is False
    assert review.sanitized_text == "ふふっ、それ、ちょっと笑えるやつだね。 そういうのあると、ちょっと楽になるよね。"
    assert "question_block_violation" in review.violation_codes()
    assert "interpretive_bright_violation" in review.violation_codes()
    assert "uncertainty_meta_violation" in review.violation_codes()


def test_review_llm_bridge_text_blocks_live_like_supportive_interpretation() -> None:
    review = review_llm_bridge_text(
        raw_text=(
            "そうなんだね。"
            "少し笑えた瞬間が、その後のしんどさがやわらぐきっかけになったんだろうな。"
            "ただの出来事というよりも、心が少し緩んだ証拠のように思える。"
        ),
        surface_context_packet=_small_shared_smile_packet(),
        fallback_text="ふふっ、それ、ちょっと笑えるやつだね。 そういうのあると、ちょっと楽になるよね。",
    )

    assert review.ok is False
    assert review.sanitized_text == "ふふっ、それ、ちょっと笑えるやつだね。 そういうのあると、ちょっと楽になるよね。"
    assert "interpretive_bright_violation" in review.violation_codes()


def test_review_llm_bridge_text_uses_bright_fallback_to_infer_small_shared_moment() -> None:
    review = review_llm_bridge_text(
        raw_text=(
            "はい、承知いたしました。"
            "少し笑いのある出来事があったんですね。"
            "その後の様子はどんなものだったでしょうか？"
            "\n\n※推定信頼度: 0.78（中） / 不確実要因: 低"
        ),
        surface_context_packet={
            "conversation_phase": "fresh_opening",
            "constraints": {"max_questions": 0},
            "source_state": {},
            "surface_profile": {},
        },
        fallback_text="ふふっ、それ、ちょっと笑えるやつだね。 そういうのあると、ちょっと楽になるよね。",
    )

    assert review.ok is False
    assert review.sanitized_text == "ふふっ、それ、ちょっと笑えるやつだね。 そういうのあると、ちょっと楽になるよね。"
    assert "question_block_violation" in review.violation_codes()
    assert "uncertainty_meta_violation" in review.violation_codes()


def test_review_llm_bridge_text_blocks_live_runtime_interpretive_sequence() -> None:
    review = review_llm_bridge_text(
        raw_text=(
            "あとの話ですね。少し笑えたとのこと、よかった。\n"
            "その後の流れは、まだ整理中でないかもしれませんね。\n"
            "ただ、そう思えた瞬間があったなら、そこには確かに何かあったのでしょう。\n"
            "ゆっくりと、でも確実に、あの続きが見えてくるでしょう。\n\n"
            "※推定信頼度: 0.78（中） / 不確実要因: 低"
        ),
        surface_context_packet={
            "conversation_phase": "fresh_opening",
            "constraints": {"max_questions": 0},
            "source_state": {},
            "surface_profile": {},
        },
        fallback_text="ふふっ、それ、ちょっと笑えるやつだね。 そういうのあると、ちょっと楽になるよね。",
    )

    assert review.ok is False
    assert review.sanitized_text == "ふふっ、それ、ちょっと笑えるやつだね。 そういうのあると、ちょっと楽になるよね。"
    assert "interpretive_bright_violation" in review.violation_codes()
    assert "uncertainty_meta_violation" in review.violation_codes()


def test_review_llm_bridge_text_uses_reaction_contract_for_non_small_scene() -> None:
    review = review_llm_bridge_text(
        raw_text=(
            "それはどういうことですか？ "
            "今の変化には、何か静かな意味があるのかもしれません。 "
            "少しずつ整理していけるといいですね。"
        ),
        surface_context_packet={
            "conversation_phase": "fresh_opening",
            "constraints": {"max_questions": 1},
            "source_state": {},
        },
        reaction_contract={
            "scale": "medium",
            "question_budget": 0,
            "interpretation_budget": "none",
        },
        fallback_text="それ、いまはそのまま受け取っておくので十分です。",
    )

    assert review.ok is False
    assert review.sanitized_text == "それ、いまはそのまま受け取っておくので十分です。"
    assert "question_block_violation" in review.violation_codes()
    assert "interpretation_budget_violation" in review.violation_codes()


def test_review_llm_bridge_text_does_not_treat_tonokoto_as_assistant_attractor() -> None:
    review = review_llm_bridge_text(
        raw_text="昨日のこととのことですね。ふふ、少し笑える感じもあったんだね。",
        surface_context_packet=_small_shared_smile_packet(),
    )

    assert review.ok is True
    assert "assistant_attractor_violation" not in review.violation_codes()


def test_review_llm_bridge_text_does_not_treat_plain_ippo_as_interpretation() -> None:
    review = review_llm_bridge_text(
        raw_text="また明日も一歩ずつでいいですね。ふふ、少し笑える余白もある。",
        surface_context_packet=_small_shared_smile_packet(),
    )

    assert review.ok is True
    assert "interpretive_bright_violation" not in review.violation_codes()


def test_review_llm_bridge_text_detects_information_request_without_question_mark() -> None:
    review = review_llm_bridge_text(
        raw_text="それ、少しだけ教えてもらえると助かります。",
        reaction_contract={
            "scale": "small",
            "question_budget": 0,
            "interpretation_budget": "none",
        },
    )

    assert review.ok is False
    assert "question_block_violation" in review.violation_codes()


def test_review_llm_bridge_text_detects_interpretation_without_keyword_list_hit() -> None:
    review = review_llm_bridge_text(
        raw_text="つまり、それは少し気持ちが変化してきたということですね。",
        reaction_contract={
            "scale": "medium",
            "question_budget": 1,
            "interpretation_budget": "none",
        },
    )

    assert review.ok is False
    assert "interpretation_budget_violation" in review.violation_codes()


def test_review_llm_bridge_text_detects_heavy_surface_act_for_small_scale() -> None:
    review = review_llm_bridge_text(
        raw_text="今は深く考えずに、ゆっくり休んでみてください。",
        reaction_contract={
            "scale": "small",
            "question_budget": 1,
            "interpretation_budget": "low",
        },
    )

    assert review.ok is False
    assert "surface_scale_violation" in review.violation_codes()


def test_review_llm_bridge_text_uses_speech_act_analysis_for_information_request() -> None:
    review = review_llm_bridge_text(
        raw_text="Share the next part when ready.",
        reaction_contract={
            "scale": "medium",
            "question_budget": 0,
            "interpretation_budget": "low",
        },
        speech_act_analysis={
            "schema_version": "speech_act.v1",
            "source": "test_classifier",
            "sentences": [
                {
                    "text": "Share the next part when ready.",
                    "labels": ["information_request"],
                    "confidence": 0.91,
                }
            ],
        },
    )

    assert review.ok is False
    assert "question_block_violation" in review.violation_codes()


def test_review_llm_bridge_text_uses_speech_act_analysis_for_interpretation() -> None:
    review = review_llm_bridge_text(
        raw_text="That means the feeling changed.",
        reaction_contract={
            "scale": "medium",
            "question_budget": 1,
            "interpretation_budget": "none",
        },
        speech_act_analysis={
            "schema_version": "speech_act.v1",
            "source": "test_classifier",
            "sentences": [
                {
                    "text": "That means the feeling changed.",
                    "labels": ["interpretation"],
                    "confidence": 0.88,
                }
            ],
        },
    )

    assert review.ok is False
    assert "interpretation_budget_violation" in review.violation_codes()


def test_review_llm_bridge_text_uses_speech_act_analysis_for_small_advice() -> None:
    review = review_llm_bridge_text(
        raw_text="Take a slow breath and organize it.",
        reaction_contract={
            "scale": "small",
            "question_budget": 1,
            "interpretation_budget": "low",
        },
        speech_act_analysis={
            "schema_version": "speech_act.v1",
            "source": "test_classifier",
            "sentences": [
                {
                    "text": "Take a slow breath and organize it.",
                    "labels": ["advice_or_directive"],
                    "confidence": 0.94,
                }
            ],
        },
    )

    assert review.ok is False
    assert "surface_scale_violation" in review.violation_codes()
