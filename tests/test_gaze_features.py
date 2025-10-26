# -*- coding: utf-8 -*-

from emot_terrain_lab.sense.gaze import extract_gaze_features


def test_extract_gaze_features_returns_summary_and_envelope() -> None:
    payload = {
        "fixation": {"target_id": "obj_42", "duration_ms": 820, "confidence": 0.8},
        "saccade_rate_hz": 2.4,
        "blink_rate_hz": 1.2,
        "pupil_z": 0.6,
        "mutual_gaze": 0.7,
        "gaze_on_me": 0.65,
        "cone_width_deg": 4.0,
    }
    cfg = {"max_fix_ms": 1200.0}
    result = extract_gaze_features(payload, cfg, t_tau=1.0)
    assert result is not None
    envelope = result["envelope"]
    summary = result["summary"]
    assert envelope.modality == "vision_gaze"
    assert "fixation_strength" in envelope.features
    assert summary.target_id == "obj_42"
