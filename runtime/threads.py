"""Conversation thread management for Listen/Soften/Guide flows."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from emot_terrain_lab.eqcore.state import CoreState, Stance
from runtime.checkpoint import CheckpointStore


@dataclass
class ThreadSpec:
    """Metadata describing a conversational thread."""

    thread_id: str
    stance: str
    created_at: float
    last_activity: float
    history: List[dict] = field(default_factory=list)
    soft_failures: int = 0
    hard_failures: int = 0

    def touch(self) -> None:
        self.last_activity = time.time()


class ThreadManager:
    """Coordinator for Listen / Soften / Guide thread lifecycles."""

    def __init__(
        self,
        *,
        checkpoint_store: Optional[CheckpointStore[CoreState]] = None,
        idle_timeout: float = 300.0,
    ) -> None:
        self.checkpoint_store = checkpoint_store or CheckpointStore[CoreState](capacity=20)
        self.idle_timeout = idle_timeout
        self._threads: Dict[str, ThreadSpec] = {}
        self._active_id: Optional[str] = None

    def start(self, thread_id: str, stance: Stance, *, snapshot: Optional[CoreState] = None) -> ThreadSpec:
        """Start a new thread, optionally storing a checkpoint."""
        now = time.time()
        spec = ThreadSpec(
            thread_id=thread_id,
            stance=stance.mode,
            created_at=now,
            last_activity=now,
            history=[],
        )
        self._threads[thread_id] = spec
        self._active_id = thread_id
        if snapshot is not None:
            self.checkpoint_store.record(snapshot, meta={"thread_id": thread_id, "stance": stance.mode})
        return spec

    def resume(self, thread_id: str) -> Optional[ThreadSpec]:
        spec = self._threads.get(thread_id)
        if spec:
            spec.touch()
            self._active_id = thread_id
        return spec

    def append_event(self, event: dict, *, thread_id: Optional[str] = None) -> None:
        """Append an event to the active thread history."""
        target_id = thread_id or self._active_id
        if target_id is None or target_id not in self._threads:
            raise KeyError("No active thread to append to.")
        spec = self._threads[target_id]
        spec.history.append(event)
        spec.touch()

    def fail(self, *, hard: bool = False, thread_id: Optional[str] = None) -> None:
        target_id = thread_id or self._active_id
        if target_id is None or target_id not in self._threads:
            return
        spec = self._threads[target_id]
        if hard:
            spec.hard_failures += 1
        else:
            spec.soft_failures += 1
        spec.touch()

    def close(self, thread_id: Optional[str] = None) -> Optional[ThreadSpec]:
        target_id = thread_id or self._active_id
        if target_id is None:
            return None
        spec = self._threads.pop(target_id, None)
        if self._active_id == target_id:
            self._active_id = None
        return spec

    def drop_idle(self) -> List[ThreadSpec]:
        """Remove threads that exceeded the idle timeout."""
        if self.idle_timeout <= 0:
            return []
        now = time.time()
        removed: List[ThreadSpec] = []
        for thread_id, spec in list(self._threads.items()):
            if now - spec.last_activity > self.idle_timeout:
                removed.append(self._threads.pop(thread_id))
                if self._active_id == thread_id:
                    self._active_id = None
        return removed

    @property
    def active(self) -> Optional[ThreadSpec]:
        if self._active_id is None:
            return None
        return self._threads.get(self._active_id)

    def latest_checkpoint(self, steps_back: int = 1) -> Optional[CoreState]:
        record = self.checkpoint_store.rewind(steps_back)
        if record is None:
            return None
        return record.state
