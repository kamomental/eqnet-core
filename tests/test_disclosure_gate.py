# -*- coding: utf-8 -*-

from emot_terrain_lab.nlg.disclosure import decide_disclosure


def test_disclosure_levels_follow_thresholds() -> None:
    thresholds = {"warn": 0.25, "must": 0.45, "ask": 0.60}
    assert decide_disclosure(0.1, [], thresholds)["level"] == "none"
    assert decide_disclosure(0.3, [], thresholds)["level"] == "warn"
    assert decide_disclosure(0.5, [], thresholds)["level"] == "must"
    assert decide_disclosure(0.7, [], thresholds)["level"] == "ask"
