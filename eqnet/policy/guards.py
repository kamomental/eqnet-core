# -*- coding: utf-8 -*-
"""Guard helpers that enforce ACK and memory invariants."""

from __future__ import annotations

from typing import Optional

from .config import load_invariants

DEFAULT_SAFE_ACK = "了解、まずは受け止めたよ。"

_TAG_TO_INTENT = {
    "happy_excited": "empathize",
    "calm_positive": "empathize",
    "angry_hot": "listen",
    "angry_quiet": "listen",
    "surprise": "clarify",
    "curious": "plan",
    "neutral": "listen",
}


def ack_intent_for_tag(tag: str) -> str:
    """Map affect tag to an intent category that policies understand."""
    return _TAG_TO_INTENT.get(tag, "listen")


def ack_guard(intent: str, will_write_memory: bool) -> bool:
    policy = load_invariants().ack
    if intent not in policy.allow_intents:
        return False
    if policy.forbid_memory_write and will_write_memory:
        return False
    return True


def enforce_ack(intent: str, text: str, will_write_memory: bool = False) -> str:
    """Return a safe ACK text that respects the guard."""
    if ack_guard(intent, will_write_memory):
        return text
    return DEFAULT_SAFE_ACK
