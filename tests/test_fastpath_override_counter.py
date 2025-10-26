# -*- coding: utf-8 -*-

from ops.nightly import _summarize_fastpath_metrics, _summarize_inner_replay_metrics


def test_fastpath_override_counter_logic() -> None:
    events = [
        {
            "meta": {
                "receipt": {
                    "fastpath": {
                        "rescue_prep": {"final_ok": True, "predicates": {"fast_rescue": True}}
                    },
                    "go_sc": {"resolution_reason": "fast_rescue_candidate"},
                },
                "resolution_reason": "go_sc_only",
            }
        }
    ]
    metrics = _summarize_fastpath_metrics(events)
    assert metrics["coverage"] == 1
    assert metrics["predicate_true"] == 1
    assert metrics["override"] == 1
    assert metrics["profiles"]["rescue_prep"]["override"] == 1


def test_inner_replay_metrics_summary() -> None:
    events = [
        {
            "meta": {
                "receipt": {
                    "inner_replay": {
                        "veto": {"score": 0.2, "decision": "execute"},
                        "u_hat": 0.5,
                        "prep": {"t_prep": 0.2, "s_max": 0.4},
                    }
                }
            }
        },
        {
            "meta": {
                "receipt": {
                    "inner_replay": {
                        "veto": {"score": 0.6, "decision": "cancel"},
                        "u_hat": -0.1,
                        "prep": {"t_prep": 0.3, "s_max": 0.5},
                    }
                }
            }
        },
    ]
    summary = _summarize_inner_replay_metrics(events)
    assert summary["count"] == 2
    assert summary["execute_rate"] == 0.5
    assert summary["cancel_rate"] == 0.5
    assert summary["cfg_top"][0][0] == "na"
