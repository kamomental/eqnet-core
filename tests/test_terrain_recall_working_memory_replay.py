import numpy as np

from emot_terrain_lab.terrain.memory import EmotionalTerrain
from emot_terrain_lab.terrain.recall import RecallEngine


class _Episodes:
    def __init__(self, episodes):
        self.episodes = episodes


class _Semantic:
    def __init__(self, patterns):
        self.patterns = patterns


def test_recall_engine_prefers_semantic_pattern_with_matching_replay_signature() -> None:
    current = np.array([0.1, 0.1, 0.0], dtype=float)
    terrain = EmotionalTerrain()
    episodic = _Episodes(
        [
            {
                "id": "ep-1",
                "emotion_pattern": {"center": [0.1, 0.1, 0.0]},
            }
        ]
    )
    semantic = _Semantic(
        [
            {
                "id": "pat-match",
                "emotion_signature": [0.1, 0.1, 0.0],
                "abstract_description": "fragile harbor promise",
                "working_memory_replay_signature": {
                    "focus": "meaning",
                    "anchor": "harbor",
                    "strength": 0.8,
                    "matched_episodes": 2,
                },
            },
            {
                "id": "pat-miss",
                "emotion_signature": [0.1, 0.1, 0.0],
                "abstract_description": "quiet garden air",
            },
        ]
    )

    engine = RecallEngine(episodic, semantic, terrain)

    results = engine.recall("harbor meaning feels unresolved", current)

    semantic_results = [item for item in results if item["layer"] == "semantic"]
    assert semantic_results[0]["item"]["id"] == "pat-match"


def test_recall_engine_uses_recurrence_weight_as_small_tiebreak() -> None:
    current = np.array([0.1, 0.1, 0.0], dtype=float)
    terrain = EmotionalTerrain()
    episodic = _Episodes([])
    semantic = _Semantic(
        [
            {
                "id": "pat-low",
                "emotion_signature": [0.1, 0.1, 0.0],
                "abstract_description": "fragile harbor promise",
                "occurrences": 2,
                "recurrence_weight": 2.0,
                "working_memory_replay_signature": {
                    "focus": "meaning",
                    "anchor": "harbor",
                    "strength": 0.8,
                    "matched_episodes": 2,
                },
            },
            {
                "id": "pat-high",
                "emotion_signature": [0.1, 0.1, 0.0],
                "abstract_description": "fragile harbor promise",
                "occurrences": 2,
                "recurrence_weight": 2.4,
                "working_memory_replay_signature": {
                    "focus": "meaning",
                    "anchor": "harbor",
                    "strength": 0.8,
                    "matched_episodes": 2,
                },
            },
        ]
    )

    engine = RecallEngine(episodic, semantic, terrain)

    results = engine.recall("harbor meaning feels unresolved", current)

    semantic_results = [item for item in results if item["layer"] == "semantic"]
    assert semantic_results[0]["item"]["id"] == "pat-high"
