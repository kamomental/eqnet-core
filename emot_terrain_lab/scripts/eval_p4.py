"""Evaluate Wave P4 replay metrics (FrameMap ICC / ΔU)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml


def icc_3_1(matrix: np.ndarray) -> float:
    matrix = np.asarray(matrix, dtype=float)
    n, k = matrix.shape
    mean_subject = matrix.mean(axis=1, keepdims=True)
    grand_mean = matrix.mean()
    bms = k * np.sum((mean_subject - grand_mean) ** 2) / max(n - 1, 1)
    wms = np.sum((matrix - mean_subject) ** 2) / max(n * (k - 1), 1)
    return float((bms - wms) / (bms + (k - 1) * wms + 1e-9))


def cohens_d(before: np.ndarray, after: np.ndarray) -> float:
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)
    pooled = np.sqrt(((before.var(ddof=1) + after.var(ddof=1)) / 2.0) + 1e-9)
    return float((after.mean() - before.mean()) / pooled)


def load_rows(path: Path) -> List[dict]:
    rows: List[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def group_by_scene(rows: List[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {}
    for row in rows:
        sid = row.get("scene_id")
        if sid:
            grouped.setdefault(sid, []).append(row)
    return grouped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Wave P4 replay metrics.")
    parser.add_argument("--path", type=Path, default=Path("data/logs/session.jsonl"))
    parser.add_argument("--template", type=Path, help="Optional YAML thresholds (uses key 'p4').")
    parser.add_argument("--out", type=Path, help="Write summary JSON here.")
    return parser.parse_args()


def load_template(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data.get("p4", {}) if isinstance(data, dict) else {}


def main(args: argparse.Namespace) -> int:
    rows = load_rows(args.path)
    if not rows:
        print("No rows found.")
        return 1

    thresholds = {
        "icc_min": 0.75,
        "delta_u_min": 0.5,
    }
    if args.template and args.template.exists():
        preset = load_template(args.template)
        thresholds["icc_min"] = float(preset.get("framemap_icc_min", thresholds["icc_min"]))
        thresholds["delta_u_min"] = float(preset.get("delta_u_min", thresholds["delta_u_min"]))

    grouped = group_by_scene(rows)
    pairs = []
    for entries in grouped.values():
        if len(entries) >= 2:
            pairs.append([entries[0].get("Psi", 0.0), entries[1].get("Psi", 0.0)])

    icc = float("nan")
    icc_pass = False
    if pairs:
        matrix = np.array(pairs, dtype=float)
        icc = icc_3_1(matrix)
        icc_pass = icc >= thresholds["icc_min"]
        print(f"FrameMap ICC={icc:.3f}  PASS? {icc_pass}")
    else:
        print("No repeated scenes for ICC computation.")

    pre = [row["U_before"] for row in rows if "U_before" in row]
    post = [row["U_after"] for row in rows if "U_after" in row]
    delta_d = float("nan")
    delta_pass = False
    if pre and post:
        delta_d = cohens_d(np.array(pre), np.array(post))
        delta_pass = delta_d >= thresholds["delta_u_min"]
        print(f"ΔU effect size d={delta_d:.2f}  PASS? {delta_pass}")
    else:
        print("ΔU data missing (U_before/U_after).")

    summary = {
        "icc": icc,
        "icc_pass": icc_pass,
        "delta_d": delta_d,
        "delta_pass": delta_pass,
        "thresholds": thresholds,
    }
    if args.out:
        args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))

