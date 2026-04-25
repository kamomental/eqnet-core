from inner_os.expression.speech_act_contract import (
    SPEECH_ACT_SCHEMA_VERSION,
    build_speech_act_classification_request,
    speech_act_analysis_from_dict,
)


def test_build_speech_act_classification_request_exposes_schema_and_labels() -> None:
    request = build_speech_act_classification_request("Tell me more.")

    assert "JSON only" in request["system_prompt"]
    assert SPEECH_ACT_SCHEMA_VERSION in request["user_prompt"]
    assert "information_request" in request["user_prompt"]
    assert "assistant_output" in request["user_prompt"]


def test_speech_act_analysis_from_dict_filters_unknown_labels() -> None:
    analysis = speech_act_analysis_from_dict(
        {
            "schema_version": SPEECH_ACT_SCHEMA_VERSION,
            "source": "test_classifier",
            "sentences": [
                {
                    "text": "Tell me more.",
                    "labels": ["information_request", "unknown_label"],
                    "confidence": 1.5,
                }
            ],
        }
    )

    assert analysis.schema_version == SPEECH_ACT_SCHEMA_VERSION
    assert analysis.source == "test_classifier"
    assert analysis.sentences[0].labels == ("information_request",)
    assert analysis.sentences[0].confidence == 1.0
    assert analysis.has_label("information_request")
