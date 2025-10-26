# -*- coding: utf-8 -*-
"""CLI utility to update consent preferences."""

import argparse
import json
from pathlib import Path

from terrain.ethics import ConsentPreferences, EthicsManager


def load_preferences(path: Path) -> ConsentPreferences:
    if path.exists():
        data = json.load(path.open("r", encoding="utf-8"))
        return ConsentPreferences.from_json(data)
    return ConsentPreferences.default()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=str, default="data/state9d")
    parser.add_argument("--enable-axis", action="append", default=[])
    parser.add_argument("--disable-axis", action="append", default=[])
    parser.add_argument("--store-dialogue", type=int, choices=[0, 1])
    parser.add_argument("--store-membrane", type=int, choices=[0, 1])
    parser.add_argument("--store-projection", type=int, choices=[0, 1])
    parser.add_argument("--store-field", type=int, choices=[0, 1])
    parser.add_argument("--allow-story-graph", type=int, choices=[0, 1])
    parser.add_argument("--allow-forgetting", type=int, choices=[0, 1])
    parser.add_argument("--retention-days", type=int)
    args = parser.parse_args()

    state_dir = Path(args.state)
    state_dir.mkdir(parents=True, exist_ok=True)
    consent_path = state_dir / "consent.json"
    prefs = load_preferences(consent_path)

    for axis in args.enable_axis:
        prefs.record_axes[axis] = True
    for axis in args.disable_axis:
        prefs.record_axes[axis] = False

    if args.store_dialogue is not None:
        prefs.store_dialogue = bool(args.store_dialogue)
    if args.store_membrane is not None:
        prefs.store_membrane = bool(args.store_membrane)
    if args.store_projection is not None:
        prefs.store_projection = bool(args.store_projection)
    if args.store_field is not None:
        prefs.store_field = bool(args.store_field)
    if args.allow_story_graph is not None:
        prefs.allow_story_graph = bool(args.allow_story_graph)
    if args.allow_forgetting is not None:
        prefs.allow_forgetting = bool(args.allow_forgetting)
    if args.retention_days is not None:
        prefs.retention_days = max(1, args.retention_days)

    with consent_path.open("w", encoding="utf-8") as f:
        json.dump(prefs.to_json(), f, ensure_ascii=False, indent=2)
    print(f"consent preferences updated at {consent_path}")


if __name__ == "__main__":
    main()
