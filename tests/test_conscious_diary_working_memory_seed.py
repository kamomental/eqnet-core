from datetime import datetime

from eqnet_core.memory.diary import DiaryWriter
from eqnet_core.models.conscious import (
    ConsciousEpisode,
    EmotionVector,
    ResponseRoute,
    SelfModel,
    SelfLayer,
    ValueGradient,
    WorldStateSnapshot,
)
from eqnet_core.models.talk_mode import TalkMode


def test_diary_writer_serializes_working_memory_seed_from_context_tags(tmp_path) -> None:
    writer = DiaryWriter(tmp_path / "conscious_diary.jsonl")
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
    payload = writer._serialize_episode(episode)
    assert payload["working_memory_seed"] == {
        "focus": "harbor_slope",
        "anchor": "harbor_slope",
    }
    assert payload["long_term_theme"] == {
        "focus": "harbor_slope",
        "anchor": "harbor_slope",
        "kind": "place",
        "summary": "quiet_harbor_slope_memory",
    }
