# -*- coding: utf-8 -*-
"""Snapshot storage for persona configurations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional


SNAPSHOT_PATH = Path("state/persona_snapshots.json")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


class PersonaState:
    """Simple JSON-based snapshot store."""

    def __init__(self, path: Path = SNAPSHOT_PATH) -> None:
        self.path = path
        self._store: Dict[str, Dict[str, object]] = {}
        if path.exists():
            try:
                self._store = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                self._store = {}

    def save_snapshot(self, label: str, payload: Dict[str, object]) -> None:
        _ensure_parent(self.path)
        self._store[label] = payload
        self.path.write_text(json.dumps(self._store, ensure_ascii=False, indent=2), encoding="utf-8")

    def restore(self, label: str) -> Optional[Dict[str, object]]:
        return self._store.get(label)

    def delete(self, label: str) -> None:
        if label in self._store:
            del self._store[label]
            self.path.write_text(json.dumps(self._store, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_snapshots(self) -> Dict[str, Dict[str, object]]:
        return dict(self._store)


__all__ = ["PersonaState", "SNAPSHOT_PATH"]
