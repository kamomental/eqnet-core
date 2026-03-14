from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

from inner_os.service import InnerOSService
from inner_os.integration_hooks import IntegrationHooks
from inner_os.memory_core import MemoryCore
from inner_os.simulation_transfer import SimulationTransferCore


class DemoLLM:
    """Minimal stand-in for an existing LLM application."""

    def generate(self, prompt: str, *, route: str, intensity: float) -> Dict[str, Any]:
        tone = "careful" if intensity < 0.6 else "open"
        text = f"[{route}/{tone}] {prompt.strip()}"
        return {"reply_text": text, "model": "demo-llm"}


def build_prompt(user_text: str, recall_result: Mapping[str, Any], gate_result: Mapping[str, Any]) -> str:
    memory_anchor = str((recall_result.get("recall_payload") or {}).get("memory_anchor") or "").strip()
    route = str(gate_result.get("route") or "watch")
    if memory_anchor:
        return f"Route={route}\nMemoryAnchor={memory_anchor}\nUser={user_text}"
    return f"Route={route}\nUser={user_text}"


def run_demo_turn(user_text: str, sensor_input: Mapping[str, Any]) -> Dict[str, Any]:
    memory_path = Path("logs/inner_os_demo_memory.jsonl")
    service = InnerOSService(
        IntegrationHooks(memory_core=MemoryCore(path=memory_path))
    )
    llm = DemoLLM()
    transfer = SimulationTransferCore()

    pre = service.pre_turn_update(
        {
            "user_input": {"text": user_text},
            "sensor_input": dict(sensor_input),
            "local_context": {"last_gate_context": {"valence": 0.1, "arousal": 0.2}},
            "current_state": {"current_energy": 0.78},
            "safety_bias": 0.1,
        }
    )
    recall = service.memory_recall(
        {
            "text_cue": user_text,
            "visual_cue": str(sensor_input.get("visual_cue") or ""),
            "world_cue": str(sensor_input.get("place_id") or ""),
            "current_state": pre["state"],
            "retrieval_summary": {},
        }
    )
    gate = service.response_gate(
        {
            "draft": {"text": user_text},
            "current_state": {**pre["state"], "mode": "reality", "memory_anchor": (recall.get("recall_payload") or {}).get("memory_anchor")},
            "safety_signals": {"safety_bias": pre["state"].get("safety_bias", 0.0)},
        }
    )

    prompt = build_prompt(user_text, recall, gate)
    llm_result = llm.generate(
        prompt,
        route=str(gate.get("route") or "watch"),
        intensity=float(gate.get("allowed_surface_intensity") or 0.5),
    )
    transferred_lessons = [
        lesson.to_memory_record()
        for lesson in transfer.promote([
            {
                "episode_id": "sim-demo-1",
                "summary": "pause, observe, then clarify before acting",
                "patterns": ["pause and observe when signals conflict"],
                "benefit_score": 0.8,
                "risk_score": 0.2,
                "transfer_ready": True,
            }
        ])
    ]
    post = service.post_turn_update(
        {
            "user_input": {"text": user_text},
            "output": llm_result,
            "current_state": {
                **pre["state"],
                "mode": "reality",
                "memory_anchor": (recall.get("recall_payload") or {}).get("memory_anchor"),
            },
            "memory_write_candidates": [recall.get("recall_payload") or {}],
            "transferred_lessons": transferred_lessons,
        }
    )
    return {
        "pre": pre,
        "recall": recall,
        "gate": gate,
        "llm": llm_result,
        "post": post,
    }


if __name__ == "__main__":
    result = run_demo_turn(
        "harbor signboard feels familiar",
        {
            "voice_level": 0.42,
            "body_stress_index": 0.33,
            "place_id": "harbor_market",
            "visual_cue": "signboard and slope",
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
