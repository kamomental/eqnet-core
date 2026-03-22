from datetime import datetime

from eqnet_core.memory.mosaic import MemoryMosaic
from eqnet_core.models.conscious import (
    ConsciousEpisode,
    EmotionVector,
    ResponseRoute,
    SelfLayer,
    SelfModel,
    ValueGradient,
    WorldStateSnapshot,
)
from eqnet_core.models.talk_mode import TalkMode


def test_memory_mosaic_persists_working_memory_seed_from_context_tags(tmp_path) -> None:
    mosaic = MemoryMosaic(tmp_path / "conscious_episodes.jsonl")
    episode = ConsciousEpisode(
        id="ep-1",
        timestamp=datetime(2026, 3, 15, 9, 0, 0),
        self_state=SelfModel(
            role_labels=["companion"],
            long_term_traits={},
            current_mode=TalkMode.WATCH,
            current_energy=0.8,
            attachment_to_user=0.5,
        ).snapshot(),
        world_state=WorldStateSnapshot(
            summary_text="previous framing",
            salient_entities=["signboard"],
            context_tags=[
                "talk:watch",
                "route:conscious",
                "wm_seed_focus:harbor_slope",
                "wm_seed_anchor:harbor_slope",
                "ltm_theme_focus:harbor_slope",
                "ltm_theme_anchor:harbor_slope",
                "ltm_theme_kind:place",
                "ltm_theme_summary:quiet_harbor_slope_memory",
            ],
            prediction_error=0.2,
        ),
        qualia=EmotionVector(
            valence=0.1,
            arousal=0.2,
            love=0.0,
            stress=0.1,
            value_gradient=ValueGradient(
                survival_bias=0.1,
                physiological_bias=0.0,
                social_bias=0.1,
                exploration_bias=0.0,
                attachment_bias=0.0,
            ),
        ),
        route=ResponseRoute.CONSCIOUS,
        dominant_self_layer=SelfLayer.AFFECTIVE,
    )
    mosaic.add_conscious_episode(episode)
    payloads = list(mosaic.iter_dicts())
    assert len(payloads) == 1
    assert payloads[0]["working_memory_seed"] == {
        "focus": "harbor_slope",
        "anchor": "harbor_slope",
    }
    assert payloads[0]["long_term_theme"] == {
        "focus": "harbor_slope",
        "anchor": "harbor_slope",
        "kind": "place",
        "summary": "quiet_harbor_slope_memory",
    }


def test_memory_mosaic_summarizes_recent_working_memory_seed(tmp_path) -> None:
    mosaic = MemoryMosaic(tmp_path / "conscious_episodes.jsonl")
    for idx in range(3):
        episode = ConsciousEpisode(
            id=f"ep-{idx}",
            timestamp=datetime(2026, 3, 15, 9, idx, 0),
            self_state=SelfModel(
                role_labels=["companion"],
                long_term_traits={},
                current_mode=TalkMode.WATCH,
                current_energy=0.8,
                attachment_to_user=0.5,
            ).snapshot(),
            world_state=WorldStateSnapshot(
                summary_text="previous framing",
                salient_entities=["signboard"],
                context_tags=[
                    "talk:watch",
                    "route:conscious",
                    "wm_seed_focus:harbor_slope",
                    "wm_seed_anchor:harbor_slope",
                    "ltm_theme_focus:harbor_slope",
                    "ltm_theme_anchor:harbor_slope",
                    "ltm_theme_kind:place",
                    "ltm_theme_summary:quiet_harbor_slope_memory",
                ],
                prediction_error=0.2,
            ),
            qualia=EmotionVector(
                valence=0.1,
                arousal=0.2,
                love=0.0,
                stress=0.1,
                value_gradient=ValueGradient(
                    survival_bias=0.1,
                    physiological_bias=0.0,
                    social_bias=0.1,
                    exploration_bias=0.0,
                    attachment_bias=0.0,
                ),
            ),
            route=ResponseRoute.CONSCIOUS,
            dominant_self_layer=SelfLayer.AFFECTIVE,
        )
        mosaic.add_conscious_episode(episode)
    summary = mosaic.latest_working_memory_seed(12)
    assert summary["focus"] == "harbor_slope"
    assert summary["anchor"] == "harbor_slope"
    assert summary["strength"] > 0.0


def test_memory_mosaic_summarizes_recent_long_term_theme(tmp_path) -> None:
    mosaic = MemoryMosaic(tmp_path / "conscious_episodes.jsonl")
    for idx in range(3):
        episode = ConsciousEpisode(
            id=f"theme-{idx}",
            timestamp=datetime(2026, 3, 15, 10, idx, 0),
            self_state=SelfModel(
                role_labels=["companion"],
                long_term_traits={},
                current_mode=TalkMode.WATCH,
                current_energy=0.8,
                attachment_to_user=0.5,
            ).snapshot(),
            world_state=WorldStateSnapshot(
                summary_text="quiet harbor memory",
                salient_entities=["signboard"],
                context_tags=[
                    "talk:watch",
                    "route:conscious",
                    "ltm_theme_focus:harbor_slope",
                    "ltm_theme_anchor:harbor_slope",
                    "ltm_theme_kind:place",
                    "ltm_theme_summary:quiet_harbor_slope_memory",
                ],
                prediction_error=0.2,
            ),
            qualia=EmotionVector(
                valence=0.1,
                arousal=0.2,
                love=0.0,
                stress=0.1,
                value_gradient=ValueGradient(
                    survival_bias=0.1,
                    physiological_bias=0.0,
                    social_bias=0.1,
                    exploration_bias=0.0,
                    attachment_bias=0.0,
                ),
            ),
            route=ResponseRoute.CONSCIOUS,
            dominant_self_layer=SelfLayer.AFFECTIVE,
        )
        mosaic.add_conscious_episode(episode)
    summary = mosaic.latest_long_term_theme(12)
    assert summary["focus"] == "harbor_slope"
    assert summary["anchor"] == "harbor_slope"
    assert summary["kind"] == "place"
    assert summary["summary"] == "quiet_harbor_slope_memory"
    assert summary["strength"] > 0.0
