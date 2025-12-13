from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def read_log(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fmt(row: Dict[str, Any]) -> str:
    comment = row.get("comment", "")
    return f"[t={row['t']:4.1f}] {comment}"


def main() -> None:
    base = Path("logs")
    off_path = base / "no_eqnet.jsonl"
    on_path = base / "with_eqnet.jsonl"
    if not off_path.exists() or not on_path.exists():
        raise SystemExit("Run life_sim/run_life_sim.py first to generate logs.")

    off_rows = read_log(off_path)
    on_rows = read_log(on_path)
    n = min(len(off_rows), len(on_rows))

    print("Q1: どちらが考えている？ Q2: どちらが引きずっている？ Q3: どちらが生活している？")
    print("-" * 140)
    for i in range(n):
        left = fmt(off_rows[i])
        right = fmt(on_rows[i])
        print(left.ljust(70) + " | " + right)


if __name__ == "__main__":
    main()
