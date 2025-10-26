# -*- coding: utf-8 -*-

from emot_terrain_lab.sense.gaze import GazeSummary
from emot_terrain_lab.social.interest_tracker import InterestTracker


def test_interest_tracker_increases_and_sets_phase() -> None:
    tracker = InterestTracker({"alpha": 0.7, "beta": 0.2, "gamma": 0.1, "eta": 0.1})
    summary = GazeSummary(
        target_id="obj_pudding",
        label="プリン",
        fixation_ms=900,
        saccade_rate_hz=1.2,
        blink_rate_hz=0.8,
        pupil_z=0.5,
        mutual_gaze=0.4,
        gaze_on_me=0.2,
        cone_width_deg=3.0,
        confidence=0.8,
    )
    report = tracker.update(summary, d_tau=0.5)
    assert report is not None
    assert report.object_id == "obj_pudding"
    assert 0 < report.interest <= 1
