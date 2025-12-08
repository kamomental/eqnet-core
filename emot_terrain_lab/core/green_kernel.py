"""Low-rank approximation of the EQNet Green function."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from devlife.core.body_nca import BodyNCA, BodyConfig


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



def run_impulse_response(
    steps: int = 128,
    grid_size: Tuple[int, int] = (64, 64),
    impulse_pos: Tuple[int, int] | None = None,
    impulse_mag: float = 0.1,
    channels: int = 6,
) -> Dict[str, np.ndarray]:
    """Simulate an impulse on the BodyNCA field and capture Φ/Ψ responses."""
    cfg = BodyConfig(grid_size=grid_size, channels=channels)
    body = BodyNCA(cfg)
    c, height, width = body.state.shape
    iy, ix = impulse_pos if impulse_pos is not None else (height // 2, width // 2)
    iy = int(np.clip(iy, 0, height - 1))
    ix = int(np.clip(ix, 0, width - 1))
    body.state[0, iy, ix] += float(impulse_mag)

    phi_local = np.zeros(steps, dtype=np.float32)
    psi_local = np.zeros(steps, dtype=np.float32)
    phi_mean = np.zeros(steps, dtype=np.float32)
    psi_mean = np.zeros(steps, dtype=np.float32)

    zero_act = np.zeros_like(body.state)
    for t in range(steps):
        obs = body.step(zero_act)
        channels_state = obs["channels"]
        phi = channels_state[0]
        psi = channels_state[1] if channels_state.shape[0] > 1 else channels_state[0]
        phi_local[t] = float(phi[iy, ix])
        psi_local[t] = float(psi[iy, ix])
        phi_mean[t] = float(phi.mean())
        psi_mean[t] = float(psi.mean())

    return {
        "phi_local": phi_local,
        "psi_local": psi_local,
        "phi_mean": phi_mean,
        "psi_mean": psi_mean,
    }



def save_impulse_response(data: Dict[str, np.ndarray], meta: Dict[str, float], path: Path) -> None:
    payload = {
        "meta": meta,
        "series": {key: value.tolist() for key, value in data.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Φ/Ψ impulse response experiment.")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--grid_size", type=int, nargs=2, default=(64, 64), help="grid height width")
    parser.add_argument("--impulse_pos", type=int, nargs=2, default=None, help="y x position of impulse")
    parser.add_argument("--impulse_mag", type=float, default=0.1)
    parser.add_argument("--channels", type=int, default=6)
    parser.add_argument("--out", type=Path, default=Path("logs/green_impulse.json"))
    return parser.parse_args()



def _cli() -> None:
    args = _parse_args()
    impulse_pos = tuple(args.impulse_pos) if args.impulse_pos is not None else None
    result = run_impulse_response(
        steps=args.steps,
        grid_size=tuple(args.grid_size),
        impulse_pos=impulse_pos,
        impulse_mag=args.impulse_mag,
        channels=args.channels,
    )
    meta = {
        "steps": args.steps,
        "grid_size": list(args.grid_size),
        "impulse_pos": list(impulse_pos) if impulse_pos else [args.grid_size[0] // 2, args.grid_size[1] // 2],
        "impulse_mag": args.impulse_mag,
        "channels": args.channels,
    }
    save_impulse_response(result, meta, args.out)
    print(f"[green_kernel] impulse response saved to {args.out}")


if __name__ == "__main__":
    _cli()
