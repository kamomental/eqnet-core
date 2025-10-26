# -*- coding: utf-8 -*-
"""Peer consent and ACL helpers for thought communication."""

from __future__ import annotations

from typing import Any, Dict, Iterable

import yaml


def load_acl(path: str | None) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    try:
        data = yaml.safe_load(open(path, "r", encoding="utf-8"))
    except Exception:
        return {}
    peers = data.get("peers") if isinstance(data, dict) else None
    if not isinstance(peers, dict):
        return {}
    normalised: Dict[str, Dict[str, Any]] = {}
    for peer_id, cfg in peers.items():
        if not isinstance(cfg, dict):
            continue
        accept_kinds = cfg.get("accept_kinds") or []
        if isinstance(accept_kinds, str):
            accept_kinds = [accept_kinds]
        normalised[peer_id] = {
            "accept_kinds": {str(k) for k in accept_kinds},
            "accept_uncertainty": bool(cfg.get("accept_uncertainty", False)),
            "max_per_turn": int(cfg.get("max_per_turn", 6)),
        }
    return normalised


def accept_kind(acl: Dict[str, Dict[str, Any]], peer_id: str, kind: str) -> bool:
    if not acl:
        return True
    kind = str(kind)
    allow_cfg = acl.get(peer_id) or acl.get("default")
    if not allow_cfg:
        return False
    if kind == "uncertainty":
        return bool(allow_cfg.get("accept_uncertainty", False))
    allowed = allow_cfg.get("accept_kinds")
    if not allowed:
        return False
    return kind in allowed


def max_per_turn(acl: Dict[str, Dict[str, Any]], peer_id: str) -> int:
    if not acl:
        return 6
    allow_cfg = acl.get(peer_id) or acl.get("default") or {}
    return int(allow_cfg.get("max_per_turn", 6))


__all__ = ["load_acl", "accept_kind", "max_per_turn"]

