from datetime import datetime

import numpy as np

from emot_terrain_lab.terrain.emotion import AXES
from emot_terrain_lab.terrain.memory import EpisodicMemory
from emot_terrain_lab.terrain.memory import EmotionalTerrain, SemanticMemory


def test_episodic_memory_keeps_working_memory_promotion_metadata() -> None:
    memory = EpisodicMemory()
    raws = [
        {
            "id": "raw-1",
            "timestamp": datetime(2026, 3, 15, 18, 0, 0).isoformat(),
            "dialogue": "the harbor promise still feels fragile tonight",
            "emotion_vec": np.array([0.61, 0.18, 0.0], dtype=float).tolist(),
            "emotion_intensity": 0.63,
            "context": {
                "working_memory_promotion": {
                    "current_focus": "meaning",
                    "focus_anchor": "harbor slope",
                    "promotion_readiness": 0.68,
                    "autobiographical_pressure": 0.52,
                    "pending_meaning": 0.61,
                    "carryover_load": 0.55,
                }
            },
        },
        {
            "id": "raw-2",
            "timestamp": datetime(2026, 3, 15, 18, 2, 0).isoformat(),
            "dialogue": "the harbor vow still feels uncertain",
            "emotion_vec": np.array([0.6, 0.19, 0.0], dtype=float).tolist(),
            "emotion_intensity": 0.61,
            "context": {
                "working_memory_promotion": {
                    "current_focus": "meaning",
                    "focus_anchor": "harbor slope",
                    "promotion_readiness": 0.72,
                    "autobiographical_pressure": 0.57,
                    "pending_meaning": 0.58,
                    "carryover_load": 0.51,
                }
            },
        },
    ]

    memory.distill_from_raw(raws)

    assert len(memory.episodes) == 1
    promotion = memory.episodes[0]["working_memory_promotion"]
    assert promotion["source_count"] == 2
    assert promotion["dominant_focus"] == "meaning"
    assert promotion["dominant_anchor"] == "harbor slope"
    assert promotion["promotion_readiness_mean"] > 0.69


def test_semantic_memory_keeps_working_memory_signature() -> None:
    terrain = EmotionalTerrain()
    memory = SemanticMemory(terrain)
    center_a = [0.0] * len(AXES)
    center_b = [0.0] * len(AXES)
    center_a[0] = 0.61
    center_a[1] = 0.18
    center_b[0] = 0.6
    center_b[1] = 0.19
    variance = [0.0] * len(AXES)
    variance[0] = 0.01
    variance[1] = 0.01
    episodes = [
        {
            "id": "ep-1",
            "summary": "fragile harbor promise",
            "emotion_pattern": {
                "center": center_a,
                "variance": variance,
                "trajectory": [center_a],
            },
            "working_memory_promotion": {
                "dominant_focus": "meaning",
                "dominant_anchor": "harbor slope",
                "promotion_readiness_mean": 0.71,
                "autobiographical_pressure_mean": 0.54,
                "pending_meaning_mean": 0.57,
                "carryover_load_mean": 0.49,
            },
        },
        {
            "id": "ep-2",
            "summary": "uncertain harbor vow",
            "emotion_pattern": {
                "center": center_b,
                "variance": variance,
                "trajectory": [center_b],
            },
            "working_memory_promotion": {
                "dominant_focus": "meaning",
                "dominant_anchor": "harbor slope",
                "promotion_readiness_mean": 0.69,
                "autobiographical_pressure_mean": 0.52,
                "pending_meaning_mean": 0.55,
                "carryover_load_mean": 0.47,
            },
        },
    ]

    memory.abstract_from_episodes(episodes)

    assert len(memory.patterns) == 1
    signature = memory.patterns[0]["working_memory_signature"]
    assert signature["source_count"] == 2
    assert signature["dominant_focus"] == "meaning"
    assert signature["dominant_anchor"] == "harbor slope"
    assert signature["promotion_readiness_mean"] > 0.69
    theme = memory.patterns[0]["long_term_theme"]
    assert theme["focus"] == "meaning"
    assert theme["anchor"] == "harbor slope"
    assert theme["strength"] > 0.0


def test_semantic_memory_keeps_working_memory_replay_signature() -> None:
    terrain = EmotionalTerrain()
    memory = SemanticMemory(terrain)
    center_a = [0.0] * len(AXES)
    center_b = [0.0] * len(AXES)
    center_a[0] = 0.61
    center_a[1] = 0.18
    center_b[0] = 0.6
    center_b[1] = 0.19
    variance = [0.0] * len(AXES)
    variance[0] = 0.01
    variance[1] = 0.01
    episodes = [
        {
            "id": "ep-1",
            "summary": "fragile harbor promise",
            "emotion_pattern": {
                "center": center_a,
                "variance": variance,
                "trajectory": [center_a],
            },
            "working_memory_promotion": {
                "dominant_focus": "meaning",
                "dominant_anchor": "harbor slope",
            },
        },
        {
            "id": "ep-2",
            "summary": "uncertain harbor vow",
            "emotion_pattern": {
                "center": center_b,
                "variance": variance,
                "trajectory": [center_b],
            },
            "working_memory_promotion": {
                "dominant_focus": "meaning",
                "dominant_anchor": "harbor slope",
            },
        },
    ]

    memory.abstract_from_episodes(
        episodes,
        replay_carryover={
            "focus": "meaning",
            "anchor": "harbor slope",
            "strength": 0.58,
            "matched_events": 2,
        },
    )

    assert len(memory.patterns) == 1
    replay_signature = memory.patterns[0]["working_memory_replay_signature"]
    assert replay_signature["focus"] == "meaning"
    assert replay_signature["anchor"] == "harbor slope"
    assert replay_signature["matched_episodes"] == 2
    assert memory.patterns[0]["recurrence_weight"] > memory.patterns[0]["occurrences"]


def test_semantic_memory_recurrence_weight_reflects_conscious_overlap() -> None:
    terrain = EmotionalTerrain()
    memory = SemanticMemory(terrain)
    center_a = [0.0] * len(AXES)
    center_b = [0.0] * len(AXES)
    center_a[0] = 0.61
    center_a[1] = 0.18
    center_b[0] = 0.6
    center_b[1] = 0.19
    variance = [0.0] * len(AXES)
    variance[0] = 0.01
    variance[1] = 0.01
    episodes = [
        {
            "id": "ep-1",
            "summary": "fragile harbor promise",
            "emotion_pattern": {
                "center": center_a,
                "variance": variance,
                "trajectory": [center_a],
            },
            "working_memory_promotion": {
                "dominant_focus": "meaning",
                "dominant_anchor": "harbor slope",
            },
        },
        {
            "id": "ep-2",
            "summary": "uncertain harbor vow",
            "emotion_pattern": {
                "center": center_b,
                "variance": variance,
                "trajectory": [center_b],
            },
            "working_memory_promotion": {
                "dominant_focus": "meaning",
                "dominant_anchor": "harbor slope",
            },
        },
    ]

    memory.abstract_from_episodes(
        episodes,
        replay_carryover={
            "focus": "meaning",
            "anchor": "harbor slope",
            "strength": 0.4,
            "matched_events": 2,
            "conscious_memory_strength": 0.5,
            "conscious_memory_overlap": 1.0,
        },
    )

    pattern = memory.patterns[0]
    replay_signature = pattern["working_memory_replay_signature"]
    assert replay_signature["conscious_memory_strength"] == 0.5
    assert replay_signature["conscious_memory_overlap"] == 1.0
    assert pattern["recurrence_weight"] > 2.25


def test_semantic_memory_recurrence_weight_reflects_long_term_theme_reinforcement() -> None:
    terrain = EmotionalTerrain()
    memory = SemanticMemory(terrain)
    center_a = [0.0] * len(AXES)
    center_b = [0.0] * len(AXES)
    center_a[0] = 0.61
    center_a[1] = 0.18
    center_b[0] = 0.6
    center_b[1] = 0.19
    variance = [0.0] * len(AXES)
    variance[0] = 0.01
    variance[1] = 0.01
    episodes = [
        {
            "id": "ep-1",
            "summary": "fragile harbor promise",
            "emotion_pattern": {"center": center_a, "variance": variance, "trajectory": [center_a]},
            "working_memory_promotion": {"dominant_focus": "meaning", "dominant_anchor": "harbor slope"},
        },
        {
            "id": "ep-2",
            "summary": "uncertain harbor vow",
            "emotion_pattern": {"center": center_b, "variance": variance, "trajectory": [center_b]},
            "working_memory_promotion": {"dominant_focus": "meaning", "dominant_anchor": "harbor slope"},
        },
    ]

    memory.abstract_from_episodes(
        episodes,
        replay_carryover={
            "focus": "meaning",
            "anchor": "harbor slope",
            "strength": 0.4,
            "matched_events": 2,
            "long_term_theme_summary": "quiet harbor slope memory",
            "long_term_theme_reinforcement": 0.32,
        },
    )

    pattern = memory.patterns[0]
    replay_signature = pattern["working_memory_replay_signature"]
    assert replay_signature["long_term_theme_summary"] == "quiet harbor slope memory"
    assert replay_signature["long_term_theme_reinforcement"] == 0.32
    assert pattern["recurrence_weight"] > 2.2
