# -*- coding: utf-8 -*-
"""Utility helpers for hashing lightweight config/dataclass payloads."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping


def cfg_fingerprint(cfg: Any) -> str:
    """
    Return a short blake2s fingerprint for a config/dataclass.

    Works with dataclasses, simple objects exposing ``__dict__``, or plain mappings.
    """

    if cfg is None:
        return "0000000000000000"
    if isinstance(cfg, Mapping):
        items = sorted(cfg.items())
    elif hasattr(cfg, "__dict__"):
        items = sorted(getattr(cfg, "__dict__").items())
    else:
        try:
            items = sorted(cfg.__iter__())  # type: ignore[attr-defined]
        except Exception:
            items = [(str(cfg), None)]
    payload = repr(items).encode("utf-8")
    return hashlib.blake2s(payload, digest_size=8).hexdigest()


__all__ = ["cfg_fingerprint"]
