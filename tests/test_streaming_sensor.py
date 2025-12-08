import numpy as np

from eqnet.hub.streaming_sensor import StreamingSensorState


def test_heart_rate_prefers_emotion_when_activity_low():
    raw = {
        "pose_vec": np.zeros(4),
        "voice_level": 0.0,
        "breath_rate": 0.0,
        "heart_rate_raw": 80.0,
        "heart_rate_baseline": 70.0,
        "inner_emotion_score": 0.8,
    }
    snapshot = StreamingSensorState.from_raw(raw)
    motion = snapshot.metrics["heart_rate_motion"]
    emotion = snapshot.metrics["heart_rate_emotion"]
    assert emotion > motion
    assert emotion > 0.0
    assert snapshot.metrics["body_state_flag"] == "overloaded"


def test_heart_rate_prefers_motion_when_activity_high():
    raw = {
        "pose_vec": np.ones(8) * 2.0,
        "voice_level": 0.7,
        "breath_rate": 0.6,
        "heart_rate_raw": 120.0,
        "heart_rate_baseline": 80.0,
        "inner_emotion_score": 0.0,
    }
    snapshot = StreamingSensorState.from_raw(raw)
    motion = snapshot.metrics["heart_rate_motion"]
    emotion = snapshot.metrics["heart_rate_emotion"]
    assert motion > emotion
    assert motion > 0.0


def test_private_flag_blocks_auto_monument_candidates():
    raw = {
        "privacy_tags": ["private"],
        "heart_rate_raw": 90.0,
        "heart_rate_baseline": 70.0,
    }
    snapshot = StreamingSensorState.from_raw(raw)
    assert snapshot.metrics["body_state_flag"] == "private_high_arousal"
    assert snapshot.metrics["body_flag_private"] == 1.0
