# -*- coding: utf-8 -*-
"""
One-touch helper that runs the main pipeline for beginners.

Usage:
    python scripts/run_quickstart.py [--logs data/logs.jsonl] [--state data/state] [--user user_000]
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path


def run_step(title: str, command: list[str], cwd: Path) -> None:
    print(f"\n=== {title} ===")
    print(" ".join(command))
    result = subprocess.run(command, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"{title} failed with exit code {result.returncode}")


def ensure_logs(root: Path, logs_path: Path) -> None:
    if logs_path.exists():
        return
    print("Log file not found. Generating sample conversations...")
    cmd = [
        sys.executable,
        "scripts/simulate_sessions.py",
        "--users",
        "3",
        "--weeks",
        "4",
        "--out",
        str(logs_path),
    ]
    run_step("Simulate sample sessions", cmd, root)


def check_dependencies() -> None:
    required = ["textual", "statsmodels", "pandas"]
    missing = []
    for pkg in required:
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("Missing packages:", ", ".join(missing))
        print("Please run `pip install -r requirements.txt` before launching the quickstart.")
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=str, default="data/logs.jsonl", help="Input conversation log (JSONL)")
    parser.add_argument("--state", type=str, default="data/state", help="State directory for outputs")
    parser.add_argument("--user", type=str, default="user_000", help="User ID to process")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    check_dependencies()
    logs_path = repo_root / args.logs
    state_path = repo_root / args.state
    state_path.mkdir(parents=True, exist_ok=True)

    ensure_logs(repo_root, logs_path)

    steps = [
        (
            "Run daily consolidation (memory + diary)",
            [
                sys.executable,
                "scripts/run_daily.py",
                "--in",
                str(logs_path),
                "--state",
                str(state_path),
                "--user",
                args.user,
            ],
        ),
        (
            "Weekly abstraction (semantic memory)",
            [
                sys.executable,
                "scripts/run_weekly.py",
                "--state",
                str(state_path),
            ],
        ),
        (
            "Export SQLite snapshot",
            [
                sys.executable,
                "scripts/export_sqlite.py",
                "--state",
                str(state_path),
                "--sqlite",
                "diary_quickstart.db",
            ],
        ),
        (
            "Export timeseries CSV",
            [
                sys.executable,
                "scripts/export_timeseries.py",
                "--state",
                str(state_path),
                "--out",
                "exports/timeseries_quickstart.csv",
            ],
        ),
        (
            "Quicklook plot",
            [
                sys.executable,
                "scripts/plot_quicklook.py",
                "--state",
                str(state_path),
                "--out",
                "figures/sample/quicklook.png",
            ],
        ),
        (
            "Granger causality analysis",
            [
                sys.executable,
                "scripts/granger_analysis.py",
                "--csv",
                "exports/timeseries_quickstart.csv",
                "--out",
                "exports/granger_quickstart.json",
            ],
        ),
        (
            "Impulse response analysis",
            [
                sys.executable,
                "scripts/impulse_response.py",
                "--csv",
                "exports/timeseries_quickstart.csv",
                "--lag",
                "1",
                "--horizon",
                "7",
                "--out",
                "exports/irf_quickstart.json",
            ],
        ),
    ]

    for title, cmd in steps:
        run_step(title, cmd, repo_root)

    diary_file = state_path / "diary.json"
    rest_file = state_path / "rest_state.json"
    print("\n=== Quickstart complete ===")
    print(f"- Diary entries: {diary_file}")
    print(f"- Rest history:  {rest_file}")
    print("- SQLite export: diary_quickstart.db")
    print("- CSV export:    exports/timeseries_quickstart.csv")
    print("- Quicklook plot: figures/sample/quicklook.png")
    print("- Granger report: exports/granger_quickstart.json")
    print("- IRF report:     exports/irf_quickstart.json")
    print("To browse diary entries interactively, run:")
    print(f"  python scripts/diary_viewer.py --state {args.state}")


if __name__ == "__main__":
    main()
