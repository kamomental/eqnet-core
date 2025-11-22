# -*- coding: utf-8 -*-
"""Prospective Drive Core = Self-Consistent Future Memory Module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


Array = np.ndarray


def _normalize(vector: Array, eps: float = 1e-8) -> Array:
    """Return an L2-normalised copy (zeros if norm is tiny)."""
    arr = np.asarray(vector, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm < eps:
        return np.zeros_like(arr)
    return arr / norm


def _cos_sim(a: Array, b: Array, eps: float = 1e-8) -> float:
    """Cosine similarity that is robust to zero vectors."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@dataclass
class PDCConfig:
    """Configuration for :class:`ProspectiveDriveCore`."""

    dim: int
    lambda_decay: float = 0.05
    alpha_past: float = 0.3
    beta_future: float = 0.3
    noise_std: float = 0.01
    base_temperature: float = 0.65
    temperature_gain: float = 0.15  # scales how E_story nudges temperature


@dataclass
class PDCState:
    """Container for the mutable state."""

    m_t: Array
    last_m_past: Optional[Array] = None
    last_m_future: Optional[Array] = None
    history_m: Optional[Array] = None


class ProspectiveDriveCore:
    """ "Prospective Drive Core layered on Phi/Psi/K/M."""

    def __init__(
        self,
        cfg: PDCConfig,
        proj_matrix: Optional[Array] = None,
        rng: Optional[np.random.Generator] = None,
        history_len: int = 32,
    ) -> None:
        self.cfg = cfg
        self.proj_matrix = proj_matrix
        self.rng = rng or np.random.default_rng()
        history = None
        if history_len > 0:
            history = np.zeros((history_len, cfg.dim), dtype=np.float32)
        self.state = PDCState(
            m_t=np.zeros(cfg.dim, dtype=np.float32),
            history_m=history,
        )
        self._history_idx = 0
        self._history_full = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def project_from_phi(self, phi_t: Array) -> Array:
        """Project ��(t) into the prospective mood subspace."""
        vec = np.asarray(phi_t, dtype=float).reshape(-1)
        if self.proj_matrix is not None:
            projected = self.proj_matrix @ vec
        else:
            projected = vec[: self.cfg.dim]
        return self._fit_dim(projected)

    def step(self, phi_t: Array, psi_t: Array, memory) -> Dict[str, Any]:
        """Advance the PDC state by one step."""
        phi_proj = self.project_from_phi(phi_t)
        if not np.any(self.state.m_t):
            self.state.m_t = phi_proj.astype(np.float32)

        m_past = self._call_memory(memory, "sample_success_vector", phi_t)
        m_future = self._call_memory(memory, "sample_future_template", phi_t, psi_t)

        m_past_hat = _normalize(self._fit_dim(m_past))
        m_future_hat = _normalize(self._fit_dim(m_future))

        lam = self.cfg.lambda_decay
        alpha = self.cfg.alpha_past
        beta = self.cfg.beta_future

        noise = self.cfg.noise_std * self.rng.standard_normal(self.cfg.dim)
        base = (self.state.m_t + phi_proj) * 0.5
        m_new = (1.0 - lam) * base + alpha * m_past_hat + beta * m_future_hat + noise
        self.state.m_t = m_new.astype(np.float32)
        self.state.last_m_past = m_past_hat.astype(np.float32)
        self.state.last_m_future = m_future_hat.astype(np.float32)
        self._update_history(self.state.m_t)

        E_story = _cos_sim(self.state.m_t, m_future_hat)
        temperature = self._compute_temperature(E_story)

        return {
            "m_t": self.state.m_t.copy(),
            "m_past_hat": (
                self.state.last_m_past.copy()
                if self.state.last_m_past is not None
                else np.zeros(self.cfg.dim)
            ),
            "m_future_hat": (
                self.state.last_m_future.copy()
                if self.state.last_m_future is not None
                else np.zeros(self.cfg.dim)
            ),
            "E_story": float(E_story),
            "T": temperature,
        }

    def compute_jerk_p95(self) -> float:
        """Return the 95th percentile of the second derivative of m_t."""
        history = self.state.history_m
        if history is None:
            return 0.0
        if not self._history_full and self._history_idx < 3:
            return 0.0
        if self._history_full:
            data = history
        else:
            data = history[: self._history_idx]
        vel = np.diff(data, axis=0)
        if vel.shape[0] < 2:
            return 0.0
        acc = np.diff(vel, axis=0)
        norms = np.linalg.norm(acc, axis=1)
        if norms.size == 0:
            return 0.0
        return float(np.percentile(norms, 95))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _fit_dim(self, vector: Array) -> Array:
        arr = np.zeros(self.cfg.dim, dtype=float)
        vec = np.asarray(vector, dtype=float).reshape(-1)
        limit = min(self.cfg.dim, vec.size)
        if limit > 0:
            arr[:limit] = vec[:limit]
        return arr

    def _call_memory(self, memory, method: str, *args) -> Array:
        func = getattr(memory, method, None)
        if callable(func):
            try:
                value = func(*args)
            except Exception:
                value = None
        else:
            value = None
        if value is None:
            return np.zeros(self.cfg.dim, dtype=float)
        return self._fit_dim(value)

    def _update_history(self, m_new: Array) -> None:
        history = self.state.history_m
        if history is None:
            return
        history[self._history_idx] = m_new
        self._history_idx = (self._history_idx + 1) % history.shape[0]
        if self._history_idx == 0:
            self._history_full = True

    def _compute_temperature(self, story_energy: Optional[float]) -> float:
        T = self.cfg.base_temperature
        if story_energy is not None:
            T *= max(0.2, 1.0 - self.cfg.temperature_gain * float(story_energy))
        return float(np.clip(T, 0.1, 1.5))


__all__ = ["PDCConfig", "PDCState", "ProspectiveDriveCore"]
