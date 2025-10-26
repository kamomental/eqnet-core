# -*- coding: utf-8 -*-
"""Quick smoke test for thought communication scaffolding."""

from __future__ import annotations

import json
import pathlib
import sys


def _load_config() -> dict:
    root = pathlib.Path(__file__).resolve().parents[1]
    cfg_path = root / "config" / "thought_bus.yaml"
    import yaml  # type: ignore

    if cfg_path.exists():
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        return data
    return {}


def main() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Provide a lightweight stub for norms_penalty if the real implementation is absent.
    import types

    if "emot_terrain_lab.culture.norms" not in sys.modules:
        norms_stub = types.ModuleType("emot_terrain_lab.culture.norms")

        def norms_penalty(norms: dict | None, action: str | None) -> float:  # type: ignore
            return 0.0

        def deontic_gate(plan: dict, norms_cfg: dict | None) -> dict:  # type: ignore
            return plan

        norms_stub.norms_penalty = norms_penalty  # type: ignore[attr-defined]
        norms_stub.deontic_gate = deontic_gate  # type: ignore[attr-defined]
        sys.modules["emot_terrain_lab.culture.norms"] = norms_stub

    if "emot_terrain_lab.value.model" not in sys.modules:
        value_stub = types.ModuleType("emot_terrain_lab.value.model")

        def compute_value_summary(state: dict | None, weights: dict | None = None) -> dict:  # type: ignore
            return {"total": 0.0}

        value_stub.compute_value_summary = compute_value_summary  # type: ignore[attr-defined]
        sys.modules["emot_terrain_lab.value.model"] = value_stub

    if "emot_terrain_lab.text.filler_inserter" not in sys.modules:
        filler_stub = types.ModuleType("emot_terrain_lab.text.filler_inserter")

        def insert_fillers(text: str, entries: list) -> str:  # type: ignore
            return text

        def to_placeholder(text: str, breaks: list) -> str:  # type: ignore
            return text

        def scan_protected_regions(text: str):  # type: ignore
            return [], {}

        filler_stub.insert_fillers = insert_fillers  # type: ignore[attr-defined]
        filler_stub.to_placeholder = to_placeholder  # type: ignore[attr-defined]
        filler_stub.scan_protected_regions = scan_protected_regions  # type: ignore[attr-defined]
        sys.modules["emot_terrain_lab.text.filler_inserter"] = filler_stub

    from emot_terrain_lab.hub.hub import Hub

    cfg = _load_config()
    cfg.setdefault("hub", {})
    cfg.setdefault("time", {"base_rate": 1.0})
    cfg.setdefault("thought_bus", cfg.get("thought_bus", {}))
    cfg["transformer_model"] = lambda prompt: f"[stub]{prompt}"

    hub = Hub(cfg)

    peer_packets = [
        {
            "id": "peer1_h1",
            "origin": "peer1",
            "kind": "hypothesis",
            "vec": [1.0, 0.5, 0.1],
            "entropy": 0.2,
            "ttl_tau": 2.0,
            "tags": [],
            "created_tau": 0.0,
        }
    ]

    ctx = {
        "agent_id": "self",
        "prompt": "Test thought bus.",
        "peers": [
            {"id": "peer1", "phase": 0.3, "omega": 0.01, "thoughts": peer_packets},
            {"id": "peer2", "phase": 0.7, "omega": 0.02},
        ],
        "bio_signals": {"junk_prob": 0.0},
        "junk_prob": 0.0,
        "urk_trace": None,
        "se_stats": None,
        "risk_p": 0.1,
        "mode": "supportive",
        "tts": False,
    }

    out = hub.run(ctx)
    receipt = out.get("receipt", {})
    has_thought_bus = "thought_bus" in receipt

    print("Thought bus present in receipt:", has_thought_bus)
    if has_thought_bus:
        print("Thought bus reason:", receipt["thought_bus"].get("reason"))
        print(json.dumps(receipt["thought_bus"], ensure_ascii=False, indent=2))
    if out.get("thought_packets"):
        print("Local packets:", json.dumps(out["thought_packets"], ensure_ascii=False, indent=2))
    if out.get("thought_bus_tx"):
        print("TX logs:", json.dumps(out["thought_bus_tx"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
