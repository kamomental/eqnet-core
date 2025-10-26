"""Lightweight checkpoint ring buffer for conversational state."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Generic, Iterable, List, Optional, TypeVar

T = TypeVar("T")


@dataclass
class CheckpointRecord(Generic[T]):
    """Single checkpoint capturing state snapshot and metadata."""

    timestamp: float
    state: T
    meta: dict = field(default_factory=dict)


class CheckpointStore(Generic[T]):
    """Ring buffer storing the last N checkpoints."""

    def __init__(self, capacity: int = 20) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        self.capacity = capacity
        self._records: List[Optional[CheckpointRecord[T]]] = [None] * capacity
        self._cursor = 0
        self._size = 0

    def record(self, state: T, *, meta: Optional[dict] = None, timestamp: Optional[float] = None) -> CheckpointRecord[T]:
        """Store a deep copy of the current state."""
        ts = timestamp if timestamp is not None else time.time()
        snapshot = copy.deepcopy(state)
        record = CheckpointRecord(timestamp=ts, state=snapshot, meta=meta or {})
        self._records[self._cursor] = record
        self._cursor = (self._cursor + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        return record

    def latest(self, n: int = 1) -> List[CheckpointRecord[T]]:
        """Return up to n latest checkpoints (most recent first)."""
        if n <= 0 or self._size == 0:
            return []
        n = min(n, self._size)
        out: List[CheckpointRecord[T]] = []
        idx = (self._cursor - 1) % self.capacity
        for _ in range(n):
            record = self._records[idx]
            if record is None:
                break
            out.append(record)
            idx = (idx - 1) % self.capacity
        return out

    def rewind(self, steps: int = 1) -> Optional[CheckpointRecord[T]]:
        """Return a past checkpoint without modifying the buffer."""
        if steps <= 0 or self._size == 0:
            return None
        steps = min(steps, self._size)
        idx = (self._cursor - steps) % self.capacity
        record = self._records[idx]
        return record

    def iter(self) -> Iterable[CheckpointRecord[T]]:
        """Iterate checkpoints from newest to oldest."""
        for record in self.latest(self._size):
            yield record
