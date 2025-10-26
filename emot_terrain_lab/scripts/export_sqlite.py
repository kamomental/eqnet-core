# -*- coding: utf-8 -*-
"""
Export diary and rest-state snapshots to a SQLite database.

Usage:
    python scripts/export_sqlite.py --state data/state --sqlite diary.db
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS diary_entries (
            day TEXT PRIMARY KEY,
            text TEXT,
            entropy REAL,
            enthalpy REAL,
            info_flux REAL,
            tags TEXT,
            highlights TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS rest_snapshots (
            captured_at TEXT PRIMARY KEY,
            payload TEXT
        );
        """
    )


def upsert_diary(conn: sqlite3.Connection, entries: Iterable[Dict[str, Any]]) -> None:
    sql = """
    INSERT INTO diary_entries (day, text, entropy, enthalpy, info_flux, tags, highlights, created_at, updated_at)
    VALUES (:day, :text, :entropy, :enthalpy, :info_flux, :tags, :highlights, :created_at, :updated_at)
    ON CONFLICT(day) DO UPDATE SET
        text = excluded.text,
        entropy = excluded.entropy,
        enthalpy = excluded.enthalpy,
        info_flux = excluded.info_flux,
        tags = excluded.tags,
        highlights = excluded.highlights,
        updated_at = excluded.updated_at;
    """
    now = datetime.utcnow().isoformat()
    payloads = []
    for entry in entries:
        metrics = entry.get("metrics", {})
        payloads.append(
            {
                "day": entry.get("day"),
                "text": entry.get("text", ""),
                "entropy": metrics.get("entropy"),
                "enthalpy": metrics.get("enthalpy"),
                "info_flux": metrics.get("info_flux"),
                "tags": ",".join(entry.get("tags", [])),
                "highlights": ",".join(entry.get("highlights", [])),
                "created_at": now,
                "updated_at": now,
            }
        )
    with conn:
        conn.executemany(sql, payloads)


def insert_rest_snapshot(conn: sqlite3.Connection, snapshot: Dict[str, Any]) -> None:
    sql = """
    INSERT OR REPLACE INTO rest_snapshots (captured_at, payload)
    VALUES (?, ?);
    """
    with conn:
        conn.execute(sql, (datetime.utcnow().isoformat(), json.dumps(snapshot, ensure_ascii=False)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=str, required=True, help="Path to state directory, e.g., data/state")
    parser.add_argument("--sqlite", type=str, required=True, help="SQLite database file to write")
    args = parser.parse_args()

    state_dir = Path(args.state)
    diary_path = state_dir / "diary.json"
    rest_path = state_dir / "rest_state.json"

    diary_payload = load_json(diary_path)
    rest_payload = load_json(rest_path)

    conn = sqlite3.connect(args.sqlite)
    ensure_schema(conn)

    entries = diary_payload.get("entries", [])
    if entries:
        upsert_diary(conn, entries)
    insert_rest_snapshot(conn, rest_payload)
    conn.close()

    print(f"Exported {len(entries)} diary entries and latest rest snapshot to {args.sqlite}")


if __name__ == "__main__":
    main()
