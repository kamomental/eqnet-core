# -*- coding: utf-8 -*-
"""
Merge or update community-specific slang dictionaries (tracks created_at / updated_at timestamps and cultural period labels).

Usage:
    python scripts/update_community_terms.py --file resources/community_terms.yaml --community vtuber_fandom --term yakkai --sentiment negative --notes "clingy behaviour" --period "2010-"
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import yaml


def load_terms(path: Path) -> Dict:
    if path.exists():
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {"communities": {}}


def save_terms(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="resources/community_terms.yaml")
    parser.add_argument("--community", type=str, required=True)
    parser.add_argument("--term", type=str, required=True)
    parser.add_argument("--sentiment", type=str, choices=["positive", "neutral", "negative"], default="neutral")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--period", type=str, default="", help="Era/period when the term was in active use (e.g., 2005-2010).")
    parser.add_argument("--delete", action="store_true", help="Delete the specified term instead of adding/updating.")
    parser.add_argument("--export-json", type=str, help="Optional JSON export of the merged dictionary.")
    args = parser.parse_args()

    path = Path(args.file)
    data = load_terms(path)
    communities = data.setdefault("communities", {})
    lexicon = communities.setdefault(args.community, {})

    if args.delete:
        lexicon.pop(args.term, None)
        if not lexicon:
            communities.pop(args.community, None)
        print(f"Removed term '{args.term}' from community '{args.community}'.")
    else:
        now = datetime.utcnow().isoformat() + "Z"
        entry = lexicon.get(args.term, {})
        created = entry.get("created_at", now)
        lexicon[args.term] = {
            "sentiment": args.sentiment,
            "notes": args.notes,
            "period": args.period,
            "created_at": created,
            "updated_at": now,
        }
        print(f"Updated term '{args.term}' in community '{args.community}'.")

    save_terms(path, data)

    if args.export_json:
        json_path = Path(args.export_json)
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote JSON export to {json_path}")


if __name__ == "__main__":
    main()
