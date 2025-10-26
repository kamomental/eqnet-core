"""Low-rank approximation of the EQNet Green function."""

from __future__ import annotations

import numpy as np


class LowRankGreen:
    """
    Maintains a low-rank Green function approximation.

    G(x) ≈ Σ w_i * φ_i(x)
    """

    def __init__(self, bases: np.ndarray, w_init: float = 0.0, lr: float = 0.05) -> None:
        self.B = np.asarray(bases, dtype=np.float32)
        if self.B.ndim != 3:
            raise ValueError("bases must be a 3D array shaped [k, H, W]")
        self.w = np.full((self.B.shape[0],), w_init, dtype=np.float32)
        self.lr = float(lr)

    def field(self) -> np.ndarray:
        """Return the current 2D field constructed from the bases."""
        return (self.w[:, None, None] * self.B).sum(axis=0)

    def update_local(self, impulse_xy: tuple[int, int], gain: float) -> None:
        """Soft-update weights using a local impulse at the given coordinates."""
        k, height, width = self.B.shape
        x0, y0 = impulse_xy
        x0 = int(np.clip(x0, 0, height - 1))
        y0 = int(np.clip(y0, 0, width - 1))
        contrib = self.B[:, x0, y0]
        norm = float(np.linalg.norm(contrib))
        if norm <= 1e-8:
            return
        delta = (self.lr * gain * contrib) / (norm + 1e-6)
        self.w = np.clip(self.w + delta.astype(np.float32), -2.0, 2.0)

    def spectral_radius(self) -> float:
        """Proxy for λ_max by computing the weight vector norm."""
        return float(np.linalg.norm(self.w, ord=2))
