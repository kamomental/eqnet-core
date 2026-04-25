from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE_PATH = REPO_ROOT / "config" / "eval" / "expression_context_profiles.json"


def enrich_records(
    records: list[Mapping[str, Any]],
    profiles: Mapping[str, Any],
) -> list[dict[str, Any]]:
    default_context = _mapping(profiles.get("default"))
    enriched: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        scenario = str(item.get("scenario") or "").strip()
        context = _deep_merge(
            default_context,
            _mapping(profiles.get(scenario)),
            _mapping(item.get("expression_context_state")),
        )
        item["expression_context_state"] = context
        enriched.append(item)
    return enriched


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    input_path = _resolve_path(path)
    records: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def load_profiles(path: str | Path) -> dict[str, Any]:
    profile_path = _resolve_path(path)
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("profile file must contain a JSON object")
    return payload


def write_jsonl(path: str | Path, records: list[Mapping[str, Any]]) -> Path:
    output_path = _resolve_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return output_path


def _deep_merge(*sources: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for source in sources:
        for key, value in source.items():
            if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
                merged[key] = _deep_merge(_mapping(merged[key]), value)
            else:
                merged[key] = value
    return merged


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _resolve_path(path: str | Path) -> Path:
    output_path = Path(path)
    return output_path if output_path.is_absolute() else REPO_ROOT / output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Add expression_context_state to core expression eval JSONL.",
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--profiles", default=str(DEFAULT_PROFILE_PATH))
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    records = load_jsonl(args.input_jsonl)
    profiles = load_profiles(args.profiles)
    enriched = enrich_records(records, profiles)
    output_path = write_jsonl(args.output_jsonl, enriched)
    summary = {
        "input_count": len(records),
        "output_count": len(enriched),
        "output_jsonl": str(output_path),
        "profile_count": len(profiles),
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"wrote {len(enriched)} records to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
