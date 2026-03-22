from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RelationalMood:
    future_pull: float = 0.0
    reverence: float = 0.0
    innocence: float = 0.0
    care: float = 0.0
    shared_world_pull: float = 0.0
    confidence_signal: float = 0.0


@dataclass
class NonverbalProfile:
    gaze_mode: str = "steady"
    pause_mode: str = "neutral"
    proximity_mode: str = "neutral"
    silence_mode: str = "neutral"
    gesture_mode: str = "contained"
    cues: list[str] = field(default_factory=list)


@dataclass
class SituationState:
    scene_mode: str = "co_present"
    repair_window_open: bool = False
    shared_attention: float = 0.0
    social_pressure: float = 0.0
    continuity_weight: float = 0.0
    current_phase: str = "ongoing"


@dataclass
class InteractionTrace:
    gaze_mode: str = "steady"
    pause_mode: str = "neutral"
    proximity_mode: str = "neutral"
    hesitation_tone: str = "neutral"
    shared_attention: float = 0.0
    repair_signal: float = 0.0
    cues: list[str] = field(default_factory=list)


@dataclass
class LiveInteractionRegulation:
    past_loop_pull: float = 0.0
    future_loop_pull: float = 0.0
    fantasy_loop_pull: float = 0.0
    shared_attention_active: float = 0.0
    strained_pause: float = 0.0
    repair_window_open: bool = False
    distance_expectation: str = "neutral"
    cues: list[str] = field(default_factory=list)


@dataclass
class InteractionStreamState:
    shared_attention_level: float = 0.0
    strained_pause_level: float = 0.0
    repair_window_open: bool = False
    repair_window_hold: float = 0.0
    contact_readiness: float = 0.0
    human_presence_signal: float = 0.0
    shared_attention_window: list[float] = field(default_factory=list)
    strained_pause_window: list[float] = field(default_factory=list)
    update_count: int = 0
    cues: list[str] = field(default_factory=list)
