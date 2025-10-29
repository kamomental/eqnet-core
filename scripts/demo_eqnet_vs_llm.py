#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick comparison between a plain LM Studio LLM response and EQNet-controlled output.

Usage:
    python scripts/demo_eqnet_vs_llm.py --prompt "今日は少し疲れています。"

Requirements:
    - LM Studio OpenAI-compatible server running (configurable via .env)
    - requirements-dev.txt installed
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
import sys
from typing import Any, Dict, Optional

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
EMOT_ROOT = REPO_ROOT / "emot_terrain_lab"
if EMOT_ROOT.exists() and str(EMOT_ROOT) not in sys.path:
    sys.path.append(str(EMOT_ROOT))

from emot_terrain_lab.terrain import llm as terrain_llm
from emot_terrain_lab.hub.runtime import EmotionalHubRuntime, RuntimeConfig


def load_env() -> None:
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        load_dotenv(override=True)


def ascii_bar(value: float, *, width: int = 24, lo: float = -1.0, hi: float = 1.0) -> str:
    if hi <= lo:
        hi = lo + 1.0
    clipped = max(lo, min(hi, value))
    ratio = (clipped - lo) / (hi - lo)
    filled = int(round(ratio * width))
    filled = max(0, min(width, filled))
    return "=" * filled + "." * (width - filled)


PLUTCHIK_CENTROIDS = {
    "Joy": (0.8, 0.6),
    "Trust": (0.6, 0.2),
    "Fear": (-0.4, 0.8),
    "Surprise": (0.0, 1.0),
    "Sadness": (-0.8, -0.4),
    "Disgust": (-0.6, -0.2),
    "Anger": (-0.6, 0.6),
    "Anticipation": (0.4, 0.4),
}


def plutchik_scores(valence: float, arousal: float) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for name, (vx, ax) in PLUTCHIK_CENTROIDS.items():
        dist = ((valence - vx) ** 2 + (arousal - ax) ** 2) ** 0.5
        score = max(0.0, 1.0 - min(1.5, dist) / 1.5)
        scores[name] = score
    return scores


def plain_llm_response(prompt: str) -> (Optional[str], float):
    system_prompt = os.getenv(
        "PLAIN_LLM_SYSTEM_PROMPT",
        "You are a neutral assistant. Respond succinctly without emotional coloration.",
    )
    start = time.perf_counter()
    text = terrain_llm.chat_text(system_prompt, prompt, temperature=0.7, top_p=0.95)
    latency = (time.perf_counter() - start) * 1000.0
    return text, latency


def eqnet_response(prompt: str) -> Dict[str, Any]:
    # Light-weight runtime (no EQNet core state required for demo)
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=False))
    result = runtime.step(user_text=prompt)
    return result


def render_eqnet_summary(result: Dict[str, Any]) -> str:
    lines = []
    affect = result.get("affect")
    metrics = result.get("metrics") or {}
    controls = result.get("controls")
    response = result.get("response")

    if affect is not None:
        lines.append("Perceived affect (user):")
        if response is not None and getattr(response, "text", ""):
            lines.append("  EQNet reply: " + response.text)
        else:
            lines.append("  EQNet reply: (no response generated)")
        lines.append("")
        lines.append("  - Valence : {:+.3f} [{}]".format(affect.valence, ascii_bar(affect.valence)))
        lines.append("  - Arousal : {:+.3f} [{}]".format(affect.arousal, ascii_bar(affect.arousal)))
        lines.append(f"  - Confidence: {affect.confidence:.3f}")
        plutchik = plutchik_scores(affect.valence, affect.arousal)
        ordered = sorted(plutchik.items(), key=lambda kv: kv[1], reverse=True)[:4]
        lines.append("  - Plutchik blend (user):")
        for name, score in ordered:
            lines.append(f"      {name:<12s} {score:.3f} [{ascii_bar(score, lo=0.0, hi=1.0)}]")

    if metrics:
        lines.append("")
        lines.append("EQNet internal field:")
        H = metrics.get("H")
        R = metrics.get("R")
        kappa = metrics.get("kappa")
        if H is not None or R is not None or kappa is not None:
            if H is not None:
                lines.append(f"  - Harmony (H): {H:.3f}")
            if R is not None:
                lines.append(f"  - Resonance (R): {R:.3f}")
            if kappa is not None:
                lines.append(f"  - Kappa (stabilisation): {kappa:.3f}")
        entropy = metrics.get("entropy")
        dissipation = metrics.get("dissipation")
        info_flux = metrics.get("info_flux")
        if entropy is not None or dissipation is not None or info_flux is not None:
            lines.append("  Detailed:")
            if entropy is not None:
                lines.append(f"    • Entropy   : {entropy:.4f}")
            if dissipation is not None:
                lines.append(f"    • Dissipation: {dissipation:.4f}")
            if info_flux is not None:
                lines.append(f"    • Info Flux : {info_flux:.5f}")

    if controls is not None:
        lines.append("")
        lines.append("Behavioural controls applied:")
        lines.append(f"  - Temperature : {controls.temperature:.3f}")
        lines.append(f"  - Top-p       : {controls.top_p:.3f}")
        lines.append(f"  - Pause (ms)  : {controls.pause_ms:.1f}")
        lines.append(f"  - Warmth      : {controls.warmth:.3f}")
        lines.append(f"  - Prosody E   : {controls.prosody_energy:.3f}")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Contrast plain LM Studio output vs EQNet emotional response.")
    parser.add_argument(
        "-p",
        "--prompt",
        default="最近あった嬉しかったことを聞かせて。",
        help="User prompt to send to both models.",
    )
    args = parser.parse_args()

    load_env()

    print("=== Plain LM Studio Response ===")
    vanilla, vanilla_latency = plain_llm_response(args.prompt)
    if vanilla:
        print(vanilla)
    else:
        print("(no response – check LM Studio server or .env settings)")
    print(f"[latency] plain LLM: {vanilla_latency:.1f} ms")

    print("\n=== EQNet-Controlled Response & Emotional Metrics ===")
    result = eqnet_response(args.prompt)
    summary = render_eqnet_summary(result)
    response = result.get("response")
    eq_latency = response.latency_ms if response is not None else float("nan")
    print(f"[latency] EQNet-controlled: {eq_latency:.1f} ms  | delta: {eq_latency - vanilla_latency:+.1f} ms")
    print(summary)

    print("\nTip: For a visual comparison run `python emot_terrain_lab/scripts/gradio_demo.py` and enter the same prompt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
