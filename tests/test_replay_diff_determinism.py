from __future__ import annotations

from pathlib import Path

from eqnet.runtime.replay.diff import build_diff_summary, load_diff_ranking_policy


def test_replay_diff_top_changes_order_is_deterministic() -> None:
    summary_a = {
        "config_meta": {"files": {"a.yaml": {"fingerprint": "aaa"}}},
        "aggregate": {
            "day_count": 2,
            "preventive_helped_rate": 0.20,
            "preventive_harmed_rate": 0.10,
            "realtime_unknown_rate": 0.05,
        },
    }
    summary_b = {
        "config_meta": {"files": {"a.yaml": {"fingerprint": "bbb"}}},
        "aggregate": {
            "day_count": 2,
            "preventive_helped_rate": 0.30,
            "preventive_harmed_rate": 0.20,
            "realtime_unknown_rate": 0.10,
        },
    }
    policy, meta = load_diff_ranking_policy(Path("configs/diff_ranking_policy_v0.yaml"))
    first = build_diff_summary(
        summary_a,
        summary_b,
        comparison_scope={"start_day_key": "2026-01-01"},
        ranking_policy=policy,
        ranking_policy_meta=meta,
    )
    second = build_diff_summary(
        summary_a,
        summary_b,
        comparison_scope={"start_day_key": "2026-01-01"},
        ranking_policy=policy,
        ranking_policy_meta=meta,
    )
    assert first == second
    assert "config_set_a_meta" in first
    assert "config_set_b_meta" in first
    assert (first.get("comparison_scope") or {}).get("start_day_key") == "2026-01-01"
    assert isinstance(first.get("ranking_inputs_fingerprint"), str)
    assert first.get("ranking_inputs_fingerprint")


def test_diff_ranking_priority_harmed_then_unknown_then_helped() -> None:
    summary_a = {
        "aggregate": {
            "preventive_harmed_rate": 0.10,
            "realtime_unknown_rate": 0.10,
            "sync_blocked_rate": 0.10,
            "preventive_helped_rate": 0.10,
            "sync_r_median_avg": 0.20,
        }
    }
    summary_b = {
        "aggregate": {
            "preventive_harmed_rate": 0.20,  # regression
            "realtime_unknown_rate": 0.20,   # regression
            "sync_blocked_rate": 0.20,       # regression
            "preventive_helped_rate": 0.20,  # improvement
            "sync_r_median_avg": 0.30,       # improvement
        }
    }
    policy, meta = load_diff_ranking_policy(Path("configs/diff_ranking_policy_v0.yaml"))
    out = build_diff_summary(summary_a, summary_b, ranking_policy=policy, ranking_policy_meta=meta)
    regressions = out.get("top_regressions") or []
    assert len(regressions) >= 3
    assert str(regressions[0]["metric"]) == "preventive_harmed_rate"
    assert str(regressions[1]["metric"]) == "realtime_unknown_rate"
    assert str(regressions[2]["metric"]) == "sync_blocked_rate"
    improvements = out.get("top_improvements") or []
    assert any(str(row.get("metric")) == "preventive_helped_rate" for row in improvements)
    assert any(str(row.get("metric")) == "sync_r_median_avg" for row in improvements)


def test_diff_ranking_tiebreak_is_metric_then_segment() -> None:
    summary_a = {"aggregate": {"x_harmed_rate": 0.1, "a_harmed_rate": 0.1}}
    summary_b = {"aggregate": {"x_harmed_rate": 0.2, "a_harmed_rate": 0.2}}
    policy, meta = load_diff_ranking_policy(Path("configs/diff_ranking_policy_v0.yaml"))
    out = build_diff_summary(summary_a, summary_b, ranking_policy=policy, ranking_policy_meta=meta)
    regressions = out.get("top_regressions") or []
    assert len(regressions) >= 2
    assert str(regressions[0]["metric"]) == "a_harmed_rate"
    assert str(regressions[1]["metric"]) == "x_harmed_rate"
