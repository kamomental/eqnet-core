# -*- coding: utf-8 -*-
"""
CLI utility to derive a persona YAML profile from free-form text.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from emot_terrain_lab.persona.profile_input import persona_from_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate persona profile YAML from free-form description.")
    parser.add_argument("--text", help="Inline free-form description. If omitted, --text-file or stdin is used.")
    parser.add_argument("--text-file", type=Path, help="Path to file containing the description.")
    parser.add_argument("--lang", help="Language hint such as ja-JP or en-US.")
    parser.add_argument("--out", type=Path, help="Optional output path to write YAML.")
    parser.add_argument("--no-preview", action="store_true", help="Do not print the YAML to stdout.")
    return parser.parse_args()


def load_text(args: argparse.Namespace) -> str:
    if args.text:
        return args.text
    if args.text_file:
        return args.text_file.read_text(encoding="utf-8")
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("Input text is required via --text, --text-file, or stdin.")


def main() -> None:
    args = parse_args()
    text = load_text(args)
    draft = persona_from_text(text, lang_hint=args.lang)

    if not args.no_preview:
        print(draft.to_yaml())
        if draft.notes:
            print("# notes:")
            for note in draft.notes:
                print(f"# - {note}")

    if args.out:
        draft.save(args.out)


if __name__ == "__main__":
    main()
