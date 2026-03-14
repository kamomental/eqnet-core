from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Mapping, Optional


@dataclass
class SimulationEpisode:
    episode_id: str
    summary: str
    patterns: List[str]
    benefit_score: float = 0.0
    risk_score: float = 0.0
    contradiction_with_real: bool = False
    transfer_ready: bool = False

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SimulationEpisode":
        return cls(
            episode_id=str(payload.get("episode_id") or payload.get("sim_episode_id") or "sim-episode"),
            summary=str(payload.get("summary") or payload.get("text") or "").strip(),
            patterns=[str(item).strip() for item in (payload.get("patterns") or []) if str(item).strip()],
            benefit_score=float(payload.get("benefit_score") or 0.0),
            risk_score=float(payload.get("risk_score") or 0.0),
            contradiction_with_real=bool(payload.get("contradiction_with_real", False)),
            transfer_ready=bool(payload.get("transfer_ready", False)),
        )

    def to_memory_record(self) -> Dict[str, Any]:
        return {
            "kind": "experienced_sim",
            "episode_id": self.episode_id,
            "summary": self.summary,
            "text": self.summary,
            "patterns": list(self.patterns),
            "benefit_score": float(self.benefit_score),
            "risk_score": float(self.risk_score),
            "contradiction_with_real": bool(self.contradiction_with_real),
            "transfer_ready": bool(self.transfer_ready),
            "provenance": "simulation",
        }


@dataclass
class TransferredLearning:
    lesson: str
    source_episode_id: str
    confidence: float
    policy_hint: str
    provenance: str = "simulation_transfer"

    def to_memory_record(self) -> Dict[str, Any]:
        return {
            "kind": "transferred_learning",
            "summary": self.lesson,
            "text": self.lesson,
            "memory_anchor": self.lesson,
            "policy_hint": self.policy_hint,
            "source_episode_id": self.source_episode_id,
            "confidence": float(self.confidence),
            "provenance": self.provenance,
        }


class SimulationTransferCore:
    """Promotes safe, abstract lessons from simulation into transferable memory."""

    def episode_record(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        return SimulationEpisode.from_mapping(payload).to_memory_record()

    def promote(self, episodes: Iterable[Mapping[str, Any]]) -> List[TransferredLearning]:
        accepted: List[TransferredLearning] = []
        for raw in episodes:
            episode = SimulationEpisode.from_mapping(raw)
            if not self._transfer_allowed(episode):
                continue
            for lesson in self._abstract_lessons(episode):
                accepted.append(lesson)
        return accepted

    def _transfer_allowed(self, episode: SimulationEpisode) -> bool:
        if episode.contradiction_with_real:
            return False
        if episode.risk_score > 0.55:
            return False
        if episode.benefit_score < 0.45 and not episode.transfer_ready:
            return False
        return bool(episode.patterns or episode.summary)

    def _abstract_lessons(self, episode: SimulationEpisode) -> List[TransferredLearning]:
        lessons: List[TransferredLearning] = []
        for pattern in episode.patterns:
            hint = self._policy_hint(pattern)
            lessons.append(
                TransferredLearning(
                    lesson=pattern,
                    source_episode_id=episode.episode_id,
                    confidence=min(1.0, 0.45 + episode.benefit_score * 0.4),
                    policy_hint=hint,
                )
            )
        if not lessons and episode.summary:
            summary_hint = self._policy_hint(episode.summary)
            lessons.append(
                TransferredLearning(
                    lesson=episode.summary[:160],
                    source_episode_id=episode.episode_id,
                    confidence=min(1.0, 0.35 + episode.benefit_score * 0.35),
                    policy_hint=summary_hint,
                )
            )
        return lessons

    def _policy_hint(self, text: str) -> str:
        lowered = text.lower()
        if "pause" in lowered or "wait" in lowered or "observe" in lowered:
            return "pause_and_observe_under_ambiguity"
        if "clarify" in lowered or "confirm" in lowered:
            return "gentle_clarification_before_commitment"
        if "distance" in lowered or "boundary" in lowered:
            return "preserve_boundary_before_escalation"
        return "promote_cautious_curiosity"
