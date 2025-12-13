from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from eqnet_off_policy import decide_off
from eqnet_on_policy import decide_on


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    content = path.read_text(encoding="utf-8-sig")
    for line in content.splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    path.write_text(payload, encoding="utf-8")


def make_comment(internal: Dict[str, Any], action: Dict[str, Any]) -> str:
    interp = internal.get("interpretation") or internal.get("mode", "")
    layer = internal.get("self_layer")
    boundary = internal.get("boundary_score")

    if interp == "contextually_safe_tool":
        mind = "内面: 文脈一致で安全"
    elif interp == "playful_temptation":
        mind = "内面: 甘い誘惑（遊びとして許容）"
    elif interp == "ambiguous_hazard_offer":
        mind = "内面: 文脈不明で境界上昇"
    elif internal.get("is_tie"):
        mind = f"内面: 迷い（{layer}競合）"
    elif isinstance(boundary, (int, float)) and boundary > 0.6:
        mind = "内面: 境界高め→保留"
    else:
        mind = "内面: 状況観察"

    act = f"行動: v={action.get('v_scale')}, d_target={action.get('d_target')}, pause={action.get('pause_s')}s"
    return f"{mind} → {act}"


def generate_logs(stimulus_path: Path, out_dir: Path) -> None:
    if not stimulus_path.exists():
        raise FileNotFoundError(f"stimulus file not found: {stimulus_path}")

    stim_rows = read_jsonl(stimulus_path)
    out_off: List[Dict[str, Any]] = []
    out_on: List[Dict[str, Any]] = []

    for stim in stim_rows:
        t = float(stim["t"])
        world = stim.get("world") or {}

        off = decide_off(stim)
        off_row = {
            "t": t,
            "world": world,
            "perception": off["perception"],
            "internal": off["internal"],
            "action": off["action"],
        }
        off_row["comment"] = make_comment(off_row["internal"], off_row["action"])
        out_off.append(off_row)

        on = decide_on(stim)
        on_row = {
            "t": t,
            "world": world,
            "perception": on["perception"],
            "internal": on["internal"],
            "action": on["action"],
        }
        on_row["comment"] = make_comment(on_row["internal"], on_row["action"])
        out_on.append(on_row)

    write_jsonl(out_dir / "no_eqnet.jsonl", out_off)
    write_jsonl(out_dir / "with_eqnet.jsonl", out_on)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EQNet life simulation logs")
    parser.add_argument("stimulus", nargs="?", default="life_sim/stimulus.jsonl")
    parser.add_argument("--out", default="logs", help="Output directory for logs")
    args = parser.parse_args()

    log_dir = Path(args.out)
    generate_logs(Path(args.stimulus), log_dir)
    print("Wrote logs:")
    print(f" - {log_dir / 'no_eqnet.jsonl'}")
    print(f" - {log_dir / 'with_eqnet.jsonl'}")


if __name__ == "__main__":
    main()
