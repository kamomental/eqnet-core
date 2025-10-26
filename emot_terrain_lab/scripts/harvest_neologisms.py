# -*- coding: utf-8 -*-
"""
Scan diary entries for emerging slang and surface candidates that are not in the
community lexicon yet. Suggestions can then be piped into
``scripts/update_community_terms.py``.

Usage:
    python scripts/harvest_neologisms.py \\
        --state data/state \\
        --community vtuber_fandom \\
        --out exports/neologisms.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import yaml  # noqa: E402


# Heuristic stop words for diary English phrases. Extend via --stop-word.
DEFAULT_STOPWORDS = {
    "and",
    "are",
    "been",
    "but",
    "day",
    "from",
    "have",
    "into",
    "just",
    "like",
    "more",
    "most",
    "over",
    "really",
    "some",
    "storygraph",
    "than",
    "that",
    "the",
    "their",
    "there",
    "they",
    "this",
    "very",
    "were",
    "with",
}

TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_'-]{2,}")


def load_diary_entries(state_dir: Path) -> List[Dict]:
    diary_path = state_dir / "diary.json"
    if not diary_path.exists():
        raise FileNotFoundError(f"Diary file not found: {diary_path}")
    payload = json.loads(diary_path.read_text(encoding="utf-8"))
    return payload.get("entries", [])


def load_lexicon(path: Path) -> Dict:
    if not path.exists():
        return {"communities": {}}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data.setdefault("communities", {})
    return data


def known_terms_for_community(lexicon: Dict, community: str) -> Set[str]:
    communities = lexicon.get("communities", {})
    known = communities.get(community, {})
    other = communities.get("default", {})
    # Use lower-case comparison so we catch case variants.
    return {term.lower() for term in {**known, **other}.keys()}


def tokenise(text: str) -> Iterable[str]:
    for match in TOKEN_PATTERN.finditer(text):
        yield match.group(0)


def collect_candidates(
    entries: Sequence[Dict],
    known_terms: Set[str],
    stopwords: Set[str],
) -> Dict[str, Dict]:
    counts: Counter[str] = Counter()
    examples: defaultdict[str, List[str]] = defaultdict(list)

    for entry in entries:
        text = (entry.get("text") or "").strip()
        if not text:
            continue
        lower_text = text.lower()
        for token in tokenise(text):
            normalised = token.lower()
            if normalised in stopwords or normalised in known_terms:
                continue
            counts[normalised] += 1
            if len(examples[normalised]) < 3:
                # Capture a short snippet around the token for context.
                # Keep ASCII to avoid mojibake in JSON readers.
                idx = lower_text.find(normalised)
                if idx != -1:
                    start = max(0, idx - 32)
                    end = min(len(text), idx + len(token) + 32)
                    snippet = text[start:end].replace("\n", " ").strip()
                    examples[normalised].append(snippet)

    candidates: Dict[str, Dict] = {}
    for term, count in counts.items():
        candidates[term] = {"count": int(count), "examples": examples[term]}
    return candidates


def build_output(
    candidates: Dict[str, Dict],
    min_count: int,
    state_dir: Path,
    community: str,
    lexicon_path: Path,
    sentiment: str,
) -> Dict:
    filtered = [
        {"term": term, **meta}
        for term, meta in sorted(candidates.items(), key=lambda item: (-item[1]["count"], item[0]))
        if meta["count"] >= min_count
    ]
    suggested_cmds = [
        (
            "python scripts/update_community_terms.py "
            f"--file {lexicon_path.as_posix()} "
            f"--community {community} "
            f"--term {item['term']} "
            f"--sentiment {sentiment}"
        )
        for item in filtered
    ]
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "state_dir": state_dir.as_posix(),
        "community": community,
        "lexicon": lexicon_path.as_posix(),
        "suggested_sentiment": sentiment,
        "min_count": min_count,
        "candidate_terms": filtered,
        "suggested_commands": suggested_cmds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=str, default="data/state", help="Directory containing diary.json")
    parser.add_argument("--community", type=str, default="default", help="Community bucket to compare against")
    parser.add_argument("--lexicon", type=str, default="resources/community_terms.yaml", help="Existing lexicon file")
    parser.add_argument("--out", type=str, default="exports/neologisms.json", help="Path for suggestion JSON output")
    parser.add_argument("--min-count", type=int, default=2, help="Minimum frequency for a term to be reported")
    parser.add_argument(
        "--sentiment",
        type=str,
        default="neutral",
        choices=["positive", "neutral", "negative"],
        help="Default sentiment scaffold for suggested commands",
    )
    parser.add_argument(
        "--stop-word",
        action="append",
        dest="extra_stopwords",
        default=[],
        help="Additional tokens to ignore (case-insensitive)",
    )
    args = parser.parse_args()

    state_dir = Path(args.state)
    entries = load_diary_entries(state_dir)
    lexicon_path = Path(args.lexicon)
    lexicon = load_lexicon(lexicon_path)

    stopwords = {word.lower() for word in DEFAULT_STOPWORDS}
    stopwords.update(word.lower() for word in args.extra_stopwords)

    known_terms = known_terms_for_community(lexicon, args.community)
    candidates = collect_candidates(entries, known_terms, stopwords)
    payload = build_output(
        candidates,
        args.min_count,
        state_dir,
        args.community,
        lexicon_path,
        args.sentiment,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(payload['candidate_terms'])} candidates to {out_path}")


if __name__ == "__main__":
    main()

