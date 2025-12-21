# -*- coding: utf-8 -*-
"""Future memory utilities for bidirectional replay alignment."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, MutableMapping, Optional

import numpy as np


Vector = np.ndarray


def _to_vector(mapping: Mapping[str, Any], keys: Iterable[str]) -> Optional[Vector]:
    values = []
    for key in keys:
        val = mapping.get(key)
        if val is None:
            values.append(0.0)
            continue
        try:
            values.append(float(val))
        except Exception:
            values.append(0.0)
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    if not np.any(np.isfinite(arr)):
        return None
    return arr


def _cosine(a: Vector, b: Vector) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)


@dataclass
class FutureMemorySnapshot:
    cos_past: Optional[float] = None
    cos_future: Optional[float] = None
    forward_loss: Optional[float] = None
    bi_loss: float = 0.0
    current_norm: Optional[float] = None
    past_norm: Optional[float] = None
    future_norm: Optional[float] = None
    vectors: MutableMapping[str, list[float]] | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "cos_past": self.cos_past,
            "cos_future": self.cos_future,
            "forward_loss": self.forward_loss,
            "bi_loss": self.bi_loss,
            "current_norm": self.current_norm,
            "past_norm": self.past_norm,
            "future_norm": self.future_norm,
        }
        if self.vectors is not None:
            payload["vectors"] = dict(self.vectors)
        return payload


class FutureMemoryController:
    """Track past success and future priors for bidirectional alignment."""

    _CURRENT_KEYS = ("Ignition", "delta_m", "jerk", "uncertainty", "novelty")
    _FIELD_KEYS = ("S", "H", "rho")
    _VALUE_SUMMARY_KEYS = (
        "total",
        "extrinsic",
        "novelty",
        "social",
        "coherence",
        "homeostasis",
        "qualia_fit",
        "norm_penalty",
    )

    def __init__(self, cfg: Mapping[str, Any], *, clock=None) -> None:
        replay_cfg = cfg or {}
        self.alpha = float(replay_cfg.get("alpha", 0.6))
        self.beta = float(replay_cfg.get("beta", 0.8))
        self.gamma = float(replay_cfg.get("gamma", 0.1))
        self.future_ttl_sec = float(replay_cfg.get("future_ttl_sec", 90.0))
        self.future_energy_threshold = float(replay_cfg.get("future_energy_threshold", 1.2))
        self._clock = clock or time.monotonic
        self._future_prior: Optional[Vector] = None
        self._future_expire: float = 0.0
        self._past_success: Optional[Vector] = None

    def update_future(self, plan: Mapping[str, Any]) -> None:
        """Pull future prior from plan and store with TTL."""
        candidate = None
        if "future_prior" in plan and isinstance(plan["future_prior"], Mapping):
            candidate = plan["future_prior"]
        elif "targets" in plan and isinstance(plan["targets"], Mapping):
            candidate = plan["targets"].get("future")
        elif "biofield" in plan and isinstance(plan["biofield"], Mapping):
            target = plan["biofield"].get("target")
            if isinstance(target, Mapping):
                candidate = target
        if isinstance(candidate, Mapping):
            vector = self._sanitize_future_prior(candidate)
            if vector is not None:
                self._future_prior = vector
                self._future_expire = self._clock() + self.future_ttl_sec
                return
        # expire if missing
        if self._clock() >= self._future_expire:
            self._future_prior = None

    def _sanitize_future_prior(self, mapping: Mapping[str, Any]) -> Optional[Vector]:
        vector = _to_vector(mapping, self._FIELD_KEYS + self._CURRENT_KEYS)
        if vector is None:
            return None
        energy = float(np.linalg.norm(vector))
        if energy > self.future_energy_threshold:
            return None
        return vector

    def update_past(self, best_choice: Optional[Mapping[str, Any]]) -> None:
        if not best_choice:
            return
        summary = best_choice.get("summary")
        if not isinstance(summary, Mapping):
            return
        vec = _to_vector(summary, self._VALUE_SUMMARY_KEYS)
        if vec is None:
            return
        if self._past_success is None:
            self._past_success = vec
        else:
            self._past_success = 0.8 * self._past_success + 0.2 * vec

    def snapshot(self, ctx_time: Mapping[str, Any], field_signals: Mapping[str, Any]) -> FutureMemorySnapshot:
        current_vec = self._current_vector(ctx_time, field_signals)
        past = self._past_success
        future = self._future_prior if self._clock() < self._future_expire else None

        cos_past = _cosine(current_vec, past) if (past is not None and current_vec is not None and len(current_vec) == len(past)) else None
        cos_future = _cosine(current_vec, future) if (future is not None and current_vec is not None and len(current_vec) == len(future)) else None

        bi_loss = 0.0
        if cos_past is not None:
            bi_loss -= self.alpha * cos_past
        if cos_future is not None:
            bi_loss -= self.beta * cos_future

        forward_loss = None
        if current_vec is not None and future is not None:
            diff = future - current_vec
            forward_loss = float(np.mean(diff * diff))

        vectors: MutableMapping[str, list[float]] = {}
        if current_vec is not None:
            vectors["current"] = current_vec.tolist()
        if past is not None:
            vectors["past"] = past.tolist()
        if future is not None:
            vectors["future"] = future.tolist()

        return FutureMemorySnapshot(
            cos_past=cos_past,
            cos_future=cos_future,
            forward_loss=forward_loss,
            bi_loss=float(bi_loss),
            current_norm=float(np.linalg.norm(current_vec)) if current_vec is not None else None,
            past_norm=float(np.linalg.norm(past)) if past is not None else None,
            future_norm=float(np.linalg.norm(future)) if future is not None else None,
            vectors=vectors or None,
        )

    def _current_vector(
        self,
        ctx_time: Mapping[str, Any],
        field_signals: Mapping[str, Any],
    ) -> Optional[Vector]:
        components: dict[str, float] = {}
        for key in self._FIELD_KEYS:
            value = field_signals.get(key)
            if value is not None:
                components[key] = float(value)
        for key in self._CURRENT_KEYS:
            value = ctx_time.get(key)
            if value is None and key.lower() != key:
                value = ctx_time.get(key.lower())
            if value is not None:
                try:
                    components[key] = float(value)
                except Exception:
                    continue
        if not components:
            return None
        vector = _to_vector(components, list(components.keys()))
        if vector is None:
            return None
        if not math.isfinite(float(np.linalg.norm(vector))):
            return None
        return vector

    def as_vectors(self) -> dict[str, list[float]]:
        payload: dict[str, list[float]] = {}
        if self._past_success is not None:
            payload["past"] = self._past_success.tolist()
        if self._future_prior is not None and self._clock() < self._future_expire:
            payload["future"] = self._future_prior.tolist()
        return payload

    def gamma(self) -> float:
        return self.gamma


__all__ = ["FutureMemoryController", "FutureMemorySnapshot"]
