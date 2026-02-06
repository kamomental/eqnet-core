from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, Protocol


class IdempotencyStore(Protocol):
    """Storage contract for idempotent operation execution."""

    def check_and_reserve(self, key: str) -> bool:
        """Return True when the key is newly reserved."""

    def mark_done(self, key: str, result_fingerprint: str) -> None:
        """Mark a reserved key as successfully completed."""

    def mark_failed(self, key: str, error_code: str) -> None:
        """Mark a reserved key as failed."""


@dataclass
class NoopIdempotencyStore:
    """Pass-through store used when idempotency is not enforced."""

    def check_and_reserve(self, key: str) -> bool:  # noqa: ARG002
        return True

    def mark_done(self, key: str, result_fingerprint: str) -> None:  # noqa: ARG002
        return None

    def mark_failed(self, key: str, error_code: str) -> None:  # noqa: ARG002
        return None


@dataclass
class InMemoryIdempotencyStore:
    """Simple in-memory implementation for tests and local runs."""

    _states: Dict[str, str] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def check_and_reserve(self, key: str) -> bool:
        with self._lock:
            state = self._states.get(key)
            if isinstance(state, str) and (
                state == "reserved"
                or state.startswith("done:")
            ):
                return False
            self._states[key] = "reserved"
            return True

    def mark_done(self, key: str, result_fingerprint: str) -> None:
        with self._lock:
            self._states[key] = f"done:{result_fingerprint}"

    def mark_failed(self, key: str, error_code: str) -> None:
        with self._lock:
            self._states[key] = f"failed:{error_code}"
