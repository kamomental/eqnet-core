# -*- coding: utf-8 -*-
"""Emotional terrain and hierarchical memory (raw → episodic → semantic)."""

from __future__ import annotations

import json
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN

from .emotion import AXES, AXIS_BOUNDS
from .llm import chat_text


class EmotionalTerrain:
    """Grid-based representation of multi-dimensional emotional state."""

    def __init__(self, grid_res: int = 10, dims: Optional[int] = None) -> None:
        self.grid_res = grid_res
        self.dims = dims or len(AXES)
        self.gradient_map: dict[tuple[int, ...], float] = {}
        self.visit_count: dict[tuple[int, ...], float] = {}

    def _normalise(self, vector: np.ndarray) -> np.ndarray:
        norm = vector.astype(float).copy()
        for idx in range(self.dims):
            axis = AXES[idx]
            low, high = AXIS_BOUNDS[axis]
            if np.isclose(high, low):
                norm[idx] = 0.0
            else:
                norm[idx] = (norm[idx] - low) / (high - low)
        return np.clip(norm, 0.0, 1.0)

    def _to_grid(self, vector: np.ndarray) -> tuple[int, ...]:
        norm = self._normalise(vector)
        grid = np.floor(norm * (self.grid_res - 1e-6)).astype(int)
        return tuple(grid.tolist())

    def update_trajectory(
        self,
        trajectory: Iterable[np.ndarray],
        intensity: float = 1.0,
        pred_grad: Optional[np.ndarray] = None,
        alpha: float = 0.2,
    ) -> None:
        for point in trajectory:
            grid_point = self._to_grid(point)
            visits = self.visit_count.get(grid_point, 0.0) + 1.0
            self.visit_count[grid_point] = visits
            smoothing = 1.0 / (1.0 + 0.1 * visits)
            observed = intensity * 0.01
            prior = 0.0 if pred_grad is None else float(np.linalg.norm(pred_grad)) * 0.01
            current = self.gradient_map.get(grid_point, 0.0)
            self.gradient_map[grid_point] = (1 - alpha) * (current * smoothing + observed) + alpha * prior

    def to_json(self) -> dict:
        return {
            "grid_res": self.grid_res,
            "dims": self.dims,
            "gradient_map": [{"coord": list(coord), "value": val} for coord, val in self.gradient_map.items()],
            "visit_count": [{"coord": list(coord), "value": val} for coord, val in self.visit_count.items()],
        }

    @staticmethod
    def from_json(payload: dict) -> "EmotionalTerrain":
        dims = payload.get("dims", len(AXES))
        terrain = EmotionalTerrain(payload.get("grid_res", 10), dims)
        terrain.gradient_map = {
            tuple(entry["coord"]): float(entry["value"]) for entry in payload.get("gradient_map", [])
        }
        terrain.visit_count = {
            tuple(entry["coord"]): float(entry["value"]) for entry in payload.get("visit_count", [])
        }
        return terrain


class RawExperienceMemory:
    """Short-term buffer of raw experiences."""

    def __init__(self, retention_days: int = 7) -> None:
        self.retention_days = retention_days
        self.experiences: list[dict] = []

    def add(self, dialogue: str, emotion_vec: np.ndarray, timestamp: datetime, context: Optional[dict] = None) -> None:
        context = context or {}
        record = {
            "id": str(uuid.uuid4()),
            "timestamp": timestamp.isoformat(),
            "dialogue": dialogue,
            "emotion_vec": list(map(float, emotion_vec)),
            "emotion_intensity": float(np.linalg.norm(emotion_vec)),
            "context": context,
            "access_count": 0,
        }
        self.experiences.append(record)

    def decay(self, now: Optional[datetime] = None) -> None:
        now = now or datetime.utcnow()
        cutoff = now - timedelta(days=self.retention_days)
        self.experiences = [
            exp
            for exp in self.experiences
            if datetime.fromisoformat(exp["timestamp"]) > cutoff or exp["emotion_intensity"] > 0.7
        ]

    def distillation_candidates(self, now: Optional[datetime] = None) -> list[dict]:
        now = now or datetime.utcnow()
        cutoff = now - timedelta(days=3)
        return [
            exp
            for exp in self.experiences
            if datetime.fromisoformat(exp["timestamp"]) < cutoff and exp["emotion_intensity"] > 0.4
        ]


class EpisodicMemory:
    """Cluster raw experiences into episodic summaries."""

    def __init__(self) -> None:
        self.episodes: list[dict] = []

    @staticmethod
    def _cluster_by_emotion(raw: list[dict]) -> list[list[dict]]:
        if not raw:
            return []
        vectors = np.array([entry["emotion_vec"] for entry in raw], dtype=float)
        labels = DBSCAN(eps=0.45, min_samples=2).fit(vectors).labels_
        groups: dict[int, list[dict]] = defaultdict(list)
        for idx, label in enumerate(labels):
            if label >= 0:
                groups[int(label)].append(raw[idx])
        return list(groups.values())

    @staticmethod
    def _summarize(cluster: list[dict]) -> str:
        joined = "\n".join(item["dialogue"] for item in cluster)
        system_prompt = (
            "あなたは対話の感情ニュアンスを保ったまま短く要約するアシスタントです。"
            "最終行に 120 文字以内の日本語要約を出力してください。"
        )
        user_prompt = f"以下を要約してください:\n{joined}\n要約:"
        summary = chat_text(system_prompt, user_prompt, temperature=0.3)
        if summary:
            return summary.strip().split("\n")[-1][:120]

        tokens = [token for token in joined.replace("\n", " ").split(" ") if token]
        top = sorted(set(tokens), key=tokens.count, reverse=True)[:5]
        return "要約: " + "/".join(top)

    @staticmethod
    def _pattern(cluster: list[dict]) -> dict:
        vectors = np.array([item["emotion_vec"] for item in cluster], dtype=float)
        return {
            "center": vectors.mean(axis=0).tolist(),
            "variance": vectors.var(axis=0).tolist(),
            "trajectory": vectors.tolist(),
        }

    @staticmethod
    def _aggregate_qualia(cluster: list[dict]) -> dict:
        energies = []
        magnitudes = []
        phases = []
        flows = []
        memories = []
        enthalpies = []
        for item in cluster:
            qualia = (item.get("context") or {}).get("qualia")
            if not qualia:
                continue
            energies.append(float(qualia.get("energy", 0.0)))
            magnitudes.append(float(qualia.get("magnitude", 0.0)))
            phases.append(float(qualia.get("phase", 0.0)))
            flows.append(np.array(qualia.get("flow", [0.0, 0.0]), dtype=float))
            memories.append(float(qualia.get("memory", 0.0)))
            enthalpies.append(float(qualia.get("enthalpy", 0.0)))
        if not energies:
            return {}
        flow_mean = np.mean(flows, axis=0) if flows else np.zeros(2, dtype=float)
        return {
            "energy_mean": float(np.mean(energies)),
            "energy_var": float(np.var(energies)),
            "magnitude_mean": float(np.mean(magnitudes)),
            "phase_mean": float(np.mean(phases)),
            "flow_mean": flow_mean.tolist(),
            "memory_mean": float(np.mean(memories)),
            "enthalpy_mean": float(np.mean(enthalpies)),
            "enthalpy_var": float(np.var(enthalpies)),
            "samples": len(energies),
        }

    def distill_from_raw(self, raws: list[dict]) -> None:
        for cluster in self._cluster_by_emotion(raws):
            episode = {
                "id": str(uuid.uuid4()),
                "timestamp": cluster[0]["timestamp"],
                "summary": self._summarize(cluster),
                "emotion_pattern": self._pattern(cluster),
                "source_ids": [item["id"] for item in cluster],
                "importance": float(np.mean([item["emotion_intensity"] for item in cluster])),
                "reconstructed_count": 0,
            }
            qualia_profile = self._aggregate_qualia(cluster)
            if qualia_profile:
                episode["qualia_profile"] = qualia_profile
            self.episodes.append(episode)


class SemanticMemory:
    """Abstract recurring patterns from episodic memories."""

    def __init__(self, terrain: EmotionalTerrain) -> None:
        self.terrain = terrain
        self.patterns: list[dict] = []

    @staticmethod
    def _similarity(p: dict, q: dict) -> float:
        a = np.array(p["center"], dtype=float)
        b = np.array(q["center"], dtype=float)
        numerator = float(np.dot(a, b))
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        return numerator / denominator

    def abstract_from_episodes(self, episodes: list[dict]) -> None:
        if not episodes:
            return

        groups: list[list[dict]] = []
        used: set[int] = set()
        for idx, epi in enumerate(episodes):
            if idx in used:
                continue
            cluster = [epi]
            used.add(idx)
            for jdx, other in enumerate(episodes):
                if jdx in used:
                    continue
                if self._similarity(epi["emotion_pattern"], other["emotion_pattern"]) > 0.9:
                    cluster.append(other)
                    used.add(jdx)
            groups.append(cluster)

        for group in groups:
            centers = np.array([item["emotion_pattern"]["center"] for item in group], dtype=float)
            signature = centers.mean(axis=0)
            qualia_signature = self._combine_qualia_profiles(group)
            pattern = {
                "id": str(uuid.uuid4()),
                "type": "recurring",
                "occurrences": len(group),
                "emotion_signature": signature.tolist(),
                "abstract_description": " / ".join(item["summary"] for item in group[:3]),
                "source_episodes": [item["id"] for item in group],
            }
            if qualia_signature:
                pattern["qualia_signature"] = qualia_signature
            self.patterns.append(pattern)

            grid_point = self.terrain._to_grid(signature)
            current = self.terrain.gradient_map.get(grid_point, 0.0)
            smoothing = 1.0 / (1.0 + 0.2 * len(group))
            self.terrain.gradient_map[grid_point] = current * smoothing
            self.terrain.visit_count[grid_point] = self.terrain.visit_count.get(grid_point, 0.0) + len(group)

    @staticmethod
    def _combine_qualia_profiles(group: list[dict]) -> dict:
        profiles = [ep.get("qualia_profile") for ep in group if ep.get("qualia_profile")]
        if not profiles:
            return {}
        energy_mean = np.array([prof.get("energy_mean", 0.0) for prof in profiles], dtype=float)
        energy_var = np.array([prof.get("energy_var", 0.0) for prof in profiles], dtype=float)
        magnitude_mean = np.array([prof.get("magnitude_mean", 0.0) for prof in profiles], dtype=float)
        phase_mean = np.array([prof.get("phase_mean", 0.0) for prof in profiles], dtype=float)
        memory_mean = np.array([prof.get("memory_mean", 0.0) for prof in profiles], dtype=float)
        flow_mean = np.array([prof.get("flow_mean", [0.0, 0.0]) for prof in profiles], dtype=float)
        enthalpy_mean = np.array([prof.get("enthalpy_mean", 0.0) for prof in profiles], dtype=float)
        enthalpy_var = np.array([prof.get("enthalpy_var", 0.0) for prof in profiles], dtype=float)
        samples = sum(int(prof.get("samples", 0)) for prof in profiles)
        return {
            "energy_mean": float(np.mean(energy_mean)),
            "energy_var": float(np.mean(energy_var)),
            "magnitude_mean": float(np.mean(magnitude_mean)),
            "phase_mean": float(np.mean(phase_mean)),
            "memory_mean": float(np.mean(memory_mean)),
            "flow_mean": flow_mean.mean(axis=0).tolist(),
            "enthalpy_mean": float(np.mean(enthalpy_mean)),
            "enthalpy_var": float(np.mean(enthalpy_var)),
            "samples": samples,
        }


def save_json(path: str | Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
