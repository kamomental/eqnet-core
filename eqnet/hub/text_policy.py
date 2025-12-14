from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Literal, Tuple, Dict

TextPolicy = Literal["redact", "hash", "truncate", "raw"]

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_DIGIT_RE = re.compile(r"\d")
_WS_RE = re.compile(r"\s+")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _mask_pii(text: str) -> str:
    masked = _EMAIL_RE.sub("<email>", text)
    masked = _DIGIT_RE.sub("x", masked)
    masked = _WS_RE.sub(" ", masked).strip()
    return masked


def is_ci_env() -> bool:
    return bool(os.getenv("CI") or os.getenv("GITHUB_ACTIONS"))


def apply_text_policy(
    user_text: Any,
    *,
    policy: TextPolicy,
    allow_raw_env: bool,
    truncate_chars: int = 200,
) -> Tuple[str | None, Dict[str, Any]]:
    """Return sanitized text and observation metadata for trace emission."""

    obs: Dict[str, Any] = {}
    if user_text is None:
        return None, obs

    text = str(user_text)
    obs["len_chars"] = len(text)

    if is_ci_env():
        return "<redacted>", obs

    if policy == "raw":
        if allow_raw_env:
            return text, obs
        return "<redacted>", obs

    if policy == "hash":
        obs["sha256"] = _sha256(text)
        return "<redacted>", obs

    if policy == "truncate":
        masked = _mask_pii(text)
        obs["sha256"] = _sha256(text)
        limit = max(0, int(truncate_chars))
        return masked[:limit], obs

    return "<redacted>", obs


__all__ = ["TextPolicy", "apply_text_policy", "is_ci_env"]
