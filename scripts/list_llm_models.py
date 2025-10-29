#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
List models exposed by the configured OpenAI-compatible endpoint.

The script checks .env/.env.example-compatible variables:
  - OPENAI_BASE_URL / OPENAI_API_KEY (custom endpoint)
  - LMSTUDIO_BASE_URL / LMSTUDIO_API_KEY (fallback to LM Studio)

Usage:
    python scripts/list_llm_models.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Dict, Optional

from dotenv import load_dotenv

DEFAULT_LM_BASE = "http://127.0.0.1:1234/v1"
DEFAULT_LM_KEY = "lm-studio"


def read_env() -> None:
    load_dotenv(override=True)


def _build_target() -> Optional[Dict[str, str]]:
    base = os.getenv("OPENAI_BASE_URL")
    key = os.getenv("OPENAI_API_KEY")
    if base and key:
        return {"base": base.rstrip("/"), "key": key, "label": "custom"}
    base = os.getenv("LMSTUDIO_BASE_URL", DEFAULT_LM_BASE)
    key = os.getenv("LMSTUDIO_API_KEY", DEFAULT_LM_KEY)
    return {"base": base.rstrip("/"), "key": key, "label": "lmstudio"}


def fetch_models(base: str, key: str, timeout: float = 5.0) -> Dict:
    url = f"{base}/models"
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def main() -> int:
    read_env()
    target = _build_target()
    if not target:
        print("[llm] No endpoint configured. Set OPENAI_BASE_URL/API_KEY or LMSTUDIO_BASE_URL/API_KEY.")
        return 1

    print(f"[llm] querying {target['base']}/models ({target['label']}) ...")
    try:
        payload = fetch_models(target["base"], target["key"])
    except urllib.error.URLError as exc:
        print(f"[llm] could not reach server: {exc}. Is the endpoint running?")
        return 1
    except json.JSONDecodeError as exc:
        print(f"[llm] endpoint response was not valid JSON: {exc}")
        return 1

    data = payload.get("data")
    if not isinstance(data, list):
        print(f"[llm] unexpected schema: {payload}")
        return 1

    if not data:
        print("[llm] no models reported. Load or expose a model first.")
        return 1

    print("[llm] available models:")
    for entry in data[:15]:
        ident = entry.get("id", "(unknown-id)")
        ctx = entry.get("context_length") or entry.get("context_window")
        owner = entry.get("owned_by") or entry.get("object")
        extras = []
        if ctx:
            extras.append(f"context={ctx}")
        if owner:
            extras.append(str(owner))
        suffix = f" ({', '.join(extras)})" if extras else ""
        print(f"  - {ident}{suffix}")

    if len(data) > 15:
        print(f"  ... and {len(data) - 15} more")

    model_hint = os.getenv("OPENAI_MODEL") or os.getenv("LMSTUDIO_MODEL")
    if model_hint:
        print(f"[llm] current preferred model = {model_hint}")
    else:
        print("[llm] set OPENAI_MODEL or LMSTUDIO_MODEL in your .env to pin a default.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
