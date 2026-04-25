from scripts.speech_act_classifier_eval import build_speech_act_classifier_eval


def test_speech_act_classifier_eval_reports_false_negative_for_critical_label() -> None:
    report = build_speech_act_classifier_eval(
        gold_records=[
            {
                "item_id": "case-1",
                "scenario_name": "small_shared_moment",
                "raw_text": "Share the next part when ready.",
                "sentences": [
                    {
                        "text": "Share the next part when ready.",
                        "labels": ["information_request"],
                        "confidence": 1.0,
                    }
                ],
            }
        ],
        prediction_records=[
            {
                "item_id": "case-1",
                "scenario_name": "small_shared_moment",
                "raw_text": "Share the next part when ready.",
                "run_metadata": {"classifier_model_label": "classifier-a"},
                "speech_act_analysis": {
                    "schema_version": "speech_act.v1",
                    "source": "classifier",
                    "sentences": [
                        {
                            "text": "Share the next part when ready.",
                            "labels": ["acknowledgement"],
                            "confidence": 0.7,
                        }
                    ],
                },
            }
        ],
    )

    metrics = report["label_metrics"]["information_request"]
    assert report["summary"]["pair_count"] == 1
    assert report["summary"]["critical_false_negative_total"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["recall"] == 0.0
    assert "information_request" in report["false_negative_examples"]


def test_speech_act_classifier_eval_reports_precision_and_recall() -> None:
    report = build_speech_act_classifier_eval(
        gold_records=[
            {
                "item_id": "case-1",
                "sentences": [
                    {
                        "text": "It may mean something changed.",
                        "labels": ["interpretation"],
                        "confidence": 1.0,
                    }
                ],
            },
            {
                "item_id": "case-2",
                "sentences": [
                    {
                        "text": "A small nod.",
                        "labels": ["small_shared_reaction"],
                        "confidence": 1.0,
                    }
                ],
            },
        ],
        prediction_records=[
            {
                "item_id": "case-1",
                "speech_act_analysis": {
                    "sentences": [
                        {
                            "text": "It may mean something changed.",
                            "labels": ["interpretation"],
                            "confidence": 0.8,
                        }
                    ],
                },
            },
            {
                "item_id": "case-2",
                "speech_act_analysis": {
                    "sentences": [
                        {
                            "text": "A small nod.",
                            "labels": ["interpretation"],
                            "confidence": 0.4,
                        }
                    ],
                },
            },
        ],
    )

    metrics = report["label_metrics"]["interpretation"]
    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 1.0
    assert report["confusion_matrix"]["small_shared_reaction"]["interpretation"] == 1
