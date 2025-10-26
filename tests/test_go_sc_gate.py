# -*- coding: utf-8 -*-

from ops.nightly import _apply_go_sc_gate


def _run_gate(
    events,
    cfg,
    nightly_cfg=None,
    tau_now: float = 0.0,
    baseline_ttl_tau: float = 1.0,
):
    nightly_cfg = nightly_cfg or {"ttl_budget": {"max_total_hours": 10.0}}
    return _apply_go_sc_gate(events, cfg, tau_now, baseline_ttl_tau, nightly_cfg)


def _event(meta):
    return {"meta": dict(meta), "value": {"total": 1.0}, "uncertainty": 0.5}


def test_masked_traces_respect_interference_precedence() -> None:
    cfg = {
        "gate": {"ttl_scale_range": [0.9, 1.5], "winner_min_percentile": 0.8},
        "precedence": {"interference_overrides_go_sc": True},
    }
    events = [
        _event(
            {
                "go_percentile": 0.95,
                "rarity": 0.9,
                "interference": {"action": "mask", "tau": 1.0, "mask_until_tau": 2.0},
            }
        )
    ]
    report = _run_gate(events, cfg)
    assert report["resolution"]["mask_wins"] == 1
    meta = events[0]["meta"]
    assert meta["ttl_scale"] == 1.0
    assert meta["resolution_reason"] == "mask_wins"


def test_rescue_requires_percentile_and_rarity() -> None:
    cfg = {
        "gate": {"ttl_scale_range": [0.9, 1.5], "winner_min_percentile": 0.8},
        "precedence": {
            "interference_overrides_go_sc": True,
            "rescue": {"enabled": True, "min_percentile": 0.9, "min_rarity": 0.8, "ttl_scale_on_rescue": 1.3},
        },
    }
    events = [
        _event({"go_percentile": 0.95, "rarity": 0.5, "interference": {"action": "mask", "tau": 0.0, "mask_until_tau": 1.0}}),
        _event({"go_percentile": 0.85, "rarity": 0.9, "interference": {"action": "mask", "tau": 0.0, "mask_until_tau": 1.0}}),
        _event({"go_percentile": 0.97, "rarity": 0.95, "interference": {"action": "mask", "tau": 0.0, "mask_until_tau": 1.0}}),
    ]
    _run_gate(events, cfg)
    assert events[0]["meta"]["resolution_reason"] == "mask_wins"
    assert events[1]["meta"]["resolution_reason"] == "mask_wins"
    assert events[2]["meta"]["resolution_reason"] == "rescued_by_go_sc"
    assert events[2]["meta"]["ttl_scale"] == 1.3


def test_ttl_budget_clamps_extensions() -> None:
    cfg = {
        "gate": {"ttl_scale_range": [0.8, 2.0], "winner_min_percentile": 0.2},
        "precedence": {"interference_overrides_go_sc": False},
    }
    nightly_cfg = {"ttl_budget": {"max_total_hours": 0.5}}
    events = [_event({"go_percentile": 0.95, "rarity": 0.2})]
    report = _run_gate(events, cfg, nightly_cfg=nightly_cfg)
    meta = events[0]["meta"]
    assert meta["ttl_scale"] == 1.5  # clamped by budget
    assert meta["resolution_reason"] == "ttl_budget_exhausted"
    assert report["ttl_budget"]["remaining_hours"] == 0.0


def test_hygiene_filter_blocks_extension() -> None:
    cfg = {
        "gate": {"ttl_scale_range": [0.8, 1.8], "winner_min_percentile": 0.1},
        "precedence": {"interference_overrides_go_sc": False},
    }
    nightly_cfg = {
        "ttl_budget": {"max_total_hours": 10.0},
        "hygiene": {"exclude_if_junk_prob_ge": 0.7},
    }
    events = [_event({"go_percentile": 0.95, "rarity": 0.3, "junk_prob": 0.9})]
    report = _run_gate(events, cfg, nightly_cfg=nightly_cfg)
    assert events[0]["meta"]["ttl_scale"] == 1.0
    assert events[0]["meta"]["resolution_reason"] == "hygiene_filtered"
    assert report["hygiene"]["filtered"] == 1
