from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml


def run_hub_once(cfg_path: str, gate_on: bool, gate_key: str, out_label: str, run_idx: int = 0) -> Path:
    from emot_terrain_lab.hub.hub import Hub

    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    hub = Hub(cfg)
    ctx = {
        "prompt": "迷っています。確信が持てません。どう判断すべき？（OPCテスト）",
        "qualia": {
            "uncertainty": 0.9,
            "pressure": 0.8,
            "novelty": 0.7,
            "norm_risk": 0.6,
            "dU_est": 0.3,
        },
        "uncertainty": 0.9,
        "pressure": 0.8,
        "novelty": 0.7,
        "norm_risk": 0.6,
        "dU_est": 0.3,
        gate_key: bool(gate_on),
    }
    import random
    rng = random.Random(123 + int(run_idx))

    def _jitter01(value: float, spread: float = 0.1) -> float:
        return max(0.0, min(1.0, float(value) + rng.uniform(-spread, spread)))

    ctx["uncertainty"] = _jitter01(ctx.get("uncertainty", 0.0))
    ctx["pressure"] = _jitter01(ctx.get("pressure", 0.0))
    qualia_block = ctx.get("qualia")
    if isinstance(qualia_block, dict):
        qualia_block["uncertainty"] = ctx["uncertainty"]
        qualia_block["pressure"] = ctx["pressure"]

    hub.run(ctx)
    state_path = Path("state") / "replay_memory.jsonl"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing {state_path}")
    out_path = Path("logs") / f"events_gate_{out_label}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(state_path, out_path)
    return out_path


def run_reporter(events_path: Path) -> dict:
    from tools.qualia_opc_report import summarize

    return summarize(events_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config/runtime.yaml")
    ap.add_argument("--gate-key", default="qualia_gate_enabled")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--runs", type=int, default=1, help="Hub runs per condition")
    args = ap.parse_args()

    state_path = Path("state") / "replay_memory.jsonl"
    state_path.parent.mkdir(parents=True, exist_ok=True)

    for label, gate_on in [("off", False), ("on", True)]:
        if args.reset and state_path.exists():
            state_path.unlink()

        events_path = None
        for i in range(max(1, args.runs)):
            print(f"\n=== Hub run {i+1}/{max(1, args.runs)}: Gate {label.upper()} ===")
            events_path = run_hub_once(args.cfg, gate_on=gate_on, gate_key=args.gate_key, out_label=label, run_idx=i)

        print(f"[OK] wrote {events_path}")
        print(f"=== Reporter: {label.upper()} ===")
        stats = run_reporter(events_path)
        print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
