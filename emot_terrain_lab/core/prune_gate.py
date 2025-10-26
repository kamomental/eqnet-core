"""Safety gate to keep Green function updates within safe bounds."""

from __future__ import annotations


class PruneGate:
    """Single-threshold guard for spectral radius."""

    def __init__(self, rho_max: float = 1.8) -> None:
        self.rho_max = float(rho_max)

    def check(self, rho: float) -> bool:
        """Return True when the spectral radius is inside the safe region."""
        return float(rho) <= self.rho_max
