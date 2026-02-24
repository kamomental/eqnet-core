#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Validate replay diff_summary.json with a fixed gate policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from eqnet.runtime.replay.diff_gate import evaluate_diff_gate, load_diff_gate_policy, render_gate_markdown


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    ap = argparse.ArgumentParser(description="Check replay diff gate constraints.")
    ap.add_argument("--diff-summary", required=True, type=str, help="Path to diff_summary.json")
    ap.add_argument(
        "--policy",
        default="configs/diff_gate_policy_v0.yaml",
        type=str,
        help="Path to diff gate policy yaml",
    )
    ap.add_argument("--out-md", default=None, type=str, help="Optional output markdown path")
    ap.add_argument("--out-json", default=None, type=str, help="Optional output json path")
    args = ap.parse_args()

    diff_path = Path(args.diff_summary)
    policy_path = Path(args.policy)
    payload = _read_json(diff_path)
    policy = load_diff_gate_policy(policy_path)
    result = evaluate_diff_gate(payload, policy)

    text = render_gate_markdown(result)
    if args.out_md:
        out_path = Path(args.out_md)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(text.strip())
    raise SystemExit(int(result.get("exit_code") or (0 if bool(result.get("ok")) else 2)))


if __name__ == "__main__":
    main()
