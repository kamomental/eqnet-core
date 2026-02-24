from __future__ import annotations

import json
from pathlib import Path

import pytest

from eqnet.runtime.replay.pipeline import ReplayConfig, run_replay


def _write_trace(path: Path) -> None:
    rows = [
        {
            "timestamp_ms": 1767225600000,
            "policy": {
                "observations": {
                    "hub": {
                        "delegation_mode": "shadow",
                        "delegate_status": "ok",
                        "idempotency_status": "done",
                        "memory_entropy_delta": 0.1,
                        "repair_state_after": "RECOGNIZE",
                        "repair_event": "NONE",
                        "output_control_profile": "neutral",
                        "day_key": "2026-01-01",
                        "episode_id": "ep-1",
                    }
                }
            },
            "qualia": {"observations": {"hub": {"output_control_fingerprint": "q-1"}}},
        },
        {
            "timestamp_ms": 1767312000000,
            "policy": {
                "observations": {
                    "hub": {
                        "delegation_mode": "shadow",
                        "delegate_status": "ok",
                        "idempotency_status": "done",
                        "memory_entropy_delta": 0.2,
                        "repair_state_after": "NON_BLAME",
                        "repair_event": "ACK",
                        "output_control_profile": "repair",
                        "day_key": "2026-01-02",
                        "episode_id": "ep-2",
                    }
                }
            },
            "qualia": {"observations": {"hub": {"output_control_fingerprint": "q-2"}}},
        },
    ]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def test_replay_idempotent_same_input_same_output(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    _write_trace(trace_path)
    cfg = ReplayConfig(
        trace_path=trace_path,
        start_day_key="2026-01-01",
        end_day_key="2026-01-02",
        config_root=Path("configs"),
    )
    a = run_replay(cfg)
    b = run_replay(cfg)
    assert a == b
    meta = a.get("config_meta") or {}
    files = (meta.get("files") or {})
    assert "fsm_policy_v0.yaml" in files
    assert "realtime_forecast_rules_v0.yaml" in files
    assert str((files.get("fsm_policy_v0.yaml") or {}).get("fingerprint") or "")
    aggregate = a.get("aggregate") or {}
    assert str(aggregate.get("behavior_change_sig_health_status") or "") in {"OK", "WARN", "FYI"}
    assert isinstance(aggregate.get("behavior_change_sig_health_reason_codes"), list)
    assert str(aggregate.get("behavior_change_sig_health_recommended_actions_mode") or "") == "advisory_only"
    assert str(aggregate.get("behavior_change_sig_health_recommended_actions_version") or "") == "v1"
    assert isinstance(aggregate.get("behavior_change_sig_health_recommended_actions"), list)
    assert isinstance(aggregate.get("behavior_change_sig_health_recommended_actions_scope"), list)
    assert isinstance(aggregate.get("behavior_change_sig_health_recommended_actions_target"), list)
    assert isinstance(aggregate.get("behavior_change_sig_health_recommended_actions_details"), list)
    assert str(aggregate.get("behavior_change_active_preset_latest") or "") == "default"
    assert str(aggregate.get("behavior_change_preset_source_latest") or "") == "manual"
    assert "default" in (aggregate.get("behavior_change_active_presets") or [])
    assert int(aggregate.get("behavior_change_preset_change_count_weekly") or 0) >= 0
    assert isinstance(aggregate.get("behavior_change_preset_change_reasons_topk"), list)
    assert isinstance(aggregate.get("behavior_change_harmed_rate_delta_avg_by_preset"), dict)
    assert isinstance(aggregate.get("behavior_change_reject_rate_delta_avg_by_preset"), dict)
    assert isinstance(aggregate.get("behavior_change_mix_weight_sig_effective_avg_by_preset"), dict)


def test_replay_requires_all_contract_files_in_config_set(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    _write_trace(trace_path)
    config_root = tmp_path / "configs"
    set_a = config_root / "config_sets" / "A"
    set_a.mkdir(parents=True, exist_ok=True)
    pattern = (config_root / "config_sets" / "{name}").as_posix()
    (config_root / "replay_policy_v0.yaml").write_text(
        "schema_version: replay_policy_v0\n"
        f"config_set_search_paths:\n  - \"{pattern}\"\n"
        "required_files:\n  - fsm_policy_v0.yaml\n",
        encoding="utf-8",
    )
    # Missing required file should fail.
    cfg = ReplayConfig(
        trace_path=trace_path,
        start_day_key="2026-01-01",
        end_day_key="2026-01-02",
        config_root=config_root,
        config_set="A",
    )
    with pytest.raises(FileNotFoundError):
        run_replay(cfg)
