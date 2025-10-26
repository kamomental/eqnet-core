"""QUBO utilities (quantum-inspired) with a simple simulated annealing solver.

Use-cases:
- RAG re-ranking (accuracy vs. diversity vs. budget)
- ETL selection (impact vs. redundancy vs. risk/cost under budget)

Formulation (example):
  minimize  x^T Q x
where Q combines linear and quadratic terms via binary relaxation:
  loss(x) = -gain@x + λ * x^T D x + μ * (cost@x - budget)^2

Mapping to Q:
  Q = λ D + μ cc^T + diag(-gain - 2 μ budget * c)
The constant μ*budget^2 is dropped.

This module is dependency-light (NumPy only) and designed for nightly batch use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def build_qubo(
    gain: np.ndarray,
    diversity: Optional[np.ndarray] = None,
    *,
    cost: Optional[np.ndarray] = None,
    budget: float = 0.0,
    lam: float = 0.2,
    mu: float = 0.0,
) -> np.ndarray:
    """Build Q matrix for the objective:
        minimize  -gain@x + lam * x^T D x + mu * (c@x - budget)^2

    Args:
        gain: shape (n,), larger is better.
        diversity: shape (n,n), symmetric. Penalizes similarity when positive.
        cost: shape (n,), item costs.
        budget: scalar budget.
        lam: weight for diversity term.
        mu: penalty for budget violation (>=0).
    Returns:
        Q: shape (n,n), symmetric.
    """
    g = np.asarray(gain, dtype=np.float64).reshape(-1)
    n = g.shape[0]
    D = np.asarray(diversity, dtype=np.float64) if diversity is not None else np.zeros((n, n), dtype=np.float64)
    if D.shape != (n, n):
        raise ValueError("diversity must be (n,n)")
    Q = lam * D.copy()
    # Budget penalty
    if mu > 0.0 and cost is not None:
        c = np.asarray(cost, dtype=np.float64).reshape(-1)
        if c.shape[0] != n:
            raise ValueError("cost must be length n")
        Q += mu * np.outer(c, c)
        diag = -g - 2.0 * mu * budget * c
    else:
        diag = -g
    # Add linear terms to diagonal (x_i^2 = x_i for binary x)
    Q += np.diag(diag)
    # Ensure symmetry
    Q = 0.5 * (Q + Q.T)
    return Q


@dataclass
class SASchedule:
    T_start: float = 1.0
    T_end: float = 1e-3
    steps: int = 50_000


def autoschedule_from_Q(Q: np.ndarray, *, steps: int = 50_000) -> SASchedule:
    """Heuristic temperature schedule from Q statistics.

    T_start is scaled to the median absolute entry in Q (robust),
    T_end is 1e-3 of T_start.
    """
    q = np.asarray(Q, dtype=np.float64)
    s = float(np.median(np.abs(q)))
    t0 = max(1e-6, s)
    t1 = max(1e-9, 1e-3 * t0)
    return SASchedule(T_start=t0, T_end=t1, steps=int(steps))


def solve_sa(
    Q: np.ndarray,
    *,
    schedule: SASchedule = SASchedule(),
    x0: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    patience: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    """Simulated annealing for QUBO: minimize x^T Q x over x in {0,1}^n.

    Uses single-bit flips with Metropolis acceptance and incremental ΔE updates.
    """
    rng = np.random.default_rng(seed)
    n = Q.shape[0]
    if Q.shape != (n, n):
        raise ValueError("Q must be square")
    # Initial state
    if x0 is None:
        x = rng.integers(0, 2, size=n, dtype=np.int8)
    else:
        x = np.asarray(x0, dtype=np.int8).copy()
        if x.shape[0] != n:
            raise ValueError("x0 length must match Q")
        x = (x > 0).astype(np.int8)
    # Precompute Qx and energy
    Qx = Q @ x
    E = float(x @ Qx)
    # Schedule
    T0, T1, S = float(schedule.T_start), float(schedule.T_end), int(schedule.steps)
    if S <= 0:
        return x, E
    best_E = E
    last_improve = 0
    for k in range(S):
        T = T0 * (T1 / T0) ** (k / max(1, S - 1))
        i = int(rng.integers(0, n))
        xi = int(x[i])
        # ΔE for flipping bit i in QUBO: ΔE = (1 - 2*xi) * (2*(Qx)_i - Q_ii)
        dE = (1 - 2 * xi) * (2.0 * Qx[i] - Q[i, i])
        if dE <= 0.0 or rng.random() < np.exp(-dE / max(1e-9, T)):
            # Accept
            x[i] = 1 - xi
            Qx += (1 - 2 * xi) * Q[:, i]
            E += dE
            if E < best_E:
                best_E = E
                last_improve = k
        if patience is not None and (k - last_improve) >= int(patience):
            break
    return x.astype(np.int8), float(E)


def cosine_similarity_matrix(emb: np.ndarray) -> np.ndarray:
    """Return cosine similarity matrix for shape (n,d) embeddings (safe-normalized)."""
    X = np.asarray(emb, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("emb must be 2D")
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.clip(norms, 1e-9, None)
    return Xn @ Xn.T


__all__ = ["build_qubo", "solve_sa", "SASchedule", "cosine_similarity_matrix", "autoschedule_from_Q"]
