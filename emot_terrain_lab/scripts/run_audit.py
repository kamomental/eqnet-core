# -*- coding: utf-8 -*-
"""Fast-path / Nightly audit helper.

This script is meant to be paired with quickstart. It does NOT mutate data; it
just validates configs and aggregates metrics so that nightly dashboards can be
reviewed quickly or scheduled inside CI/cron.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(title: str, command: list[str], cwd: Path, *, allow_missing: bool = False) -> None:
    print(f"\n=== {title} ===")
    print(" ".join(command))
    result = subprocess.run(command, cwd=cwd)
    if result.returncode != 0:
        if allow_missing:
            print(f"[warn] {title} failed (exit {result.returncode}) but was marked optional.")
            return
        raise RuntimeError(f"{title} failed with exit code {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fastpath-config",
        default="config/fastpath.yaml",
        help="Path to fast-path configuration YAML",
    )
    parser.add_argument(
        "--nightly-reports",
        default="reports/nightly",
        help="Directory containing nightly JSON outputs",
    )
    parser.add_argument(
        "--metrics-out",
        default="reports/fastpath_metrics_baseline.json",
        help="Aggregated fast-path metric output",
    )
    parser.add_argument(
        "--validate-nightly",
        action="store_true",
        help="Also run tools/validate_config.py on nightly-related configs if present",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    steps: list[tuple[str, list[str], bool]] = [
        (
            "Validate fast-path config",
            [sys.executable, "tools/validate_config.py", args.fastpath_config],
            False,
        ),
        (
            "Summarize fast-path metrics",
            [
                sys.executable,
                "ops/jobs/fastpath_metrics.py",
                "--reports",
                args.nightly_reports,
                "--out",
                args.metrics_out,
            ],
            True,  # allow empty nightly directories
        ),
    ]

    if args.validate_nightly:
        nightly_cfg = Path("config/nightly.yaml")
        if nightly_cfg.exists():
            steps.append(
                (
                    "Validate nightly config",
                    [sys.executable, "tools/validate_config.py", str(nightly_cfg)],
                    True,
                )
            )

    for title, cmd, allow_missing in steps:
        run_step(title, cmd, repo_root, allow_missing=allow_missing)

    print("\n=== Audit helper complete ===")
    print(f"- Fast-path config validated: {args.fastpath_config}")
    print(f"- Metrics aggregated to:      {args.metrics_out}")


if __name__ == "__main__":
    main()
