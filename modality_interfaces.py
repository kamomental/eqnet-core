from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ModalityWeights:
    vision: float = 1.0
    audio: float = 1.0
    tactile: float = 1.0
    proprio: float = 1.0
    intero: float = 1.0
    gaze: float = 1.0
    gesture: float = 1.0
    timing: float = 1.0


@dataclass
class ModalityUsage:
    vision: float = 0.0
    audio: float = 0.0
    tactile: float = 0.0
    proprio: float = 0.0
    intero: float = 0.0
    gaze: float = 0.0
    gesture: float = 0.0
    timing: float = 0.0


class ModalityDriftEngine(ABC):
    """§14〜16: Modality Drift / Growth interface."""

    @abstractmethod
    def update(self, weights: ModalityWeights, usage: ModalityUsage) -> ModalityWeights:
        raise NotImplementedError


class EmergencyOverride(ABC):
    """§17: Emergency Sensory Override interface."""

    @abstractmethod
    def apply(self, drift_weights: ModalityWeights, emergency_flag: bool) -> ModalityWeights:
        raise NotImplementedError
