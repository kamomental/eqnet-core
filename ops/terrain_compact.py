# -*- coding: utf-8 -*-
"""Terrain log compaction helpers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from textwrap import dedent


def _import_duckdb():
    try:
        import duckdb  # type: ignore
    except ImportError as exc:  # pragma: no cover - explicit guidance
        raise SystemExit(
            "duckdb Python package is required. Install with `pip install duckdb==0.9.2`."
        ) from exc
    return duckdb


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert NDJSON terrain logs into ZSTD-compressed Parquet partitions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent(
            """\
            Examples
            --------
            python -m ops.terrain_compact \\
              --input 'logs/terrain/terrain.jsonl.part.*' \\
              --output dataset/terrain_parquet \\
              --timestamp-column ts
            """
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Glob pointing at NDJSON (optionally .zst) chunks. Example: logs/terrain/*.jsonl.zst",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination directory root (will contain date=YYYY-MM-DD/part-*.parquet).",
    )
    parser.add_argument(
        "--timestamp-column",
        default="ts",
        help="Column containing ISO timestamp strings (default: ts).",
    )
    parser.add_argument(
        "--date-column",
        default="date",
        help="Name of the DATE column to generate for partitioning (default: date).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="DuckDB PRAGMA threads value. 0 lets DuckDB decide (default: 0).",
    )
    parser.add_argument(
        "--compression",
        default="ZSTD",
        choices=["ZSTD", "SNAPPY", "GZIP"],
        help="Parquet compression codec. Default: ZSTD.",
    )
    parser.add_argument(
        "--database",
        default=":memory:",
        help="DuckDB database path. Defaults to in-memory.",
    )
    return parser.parse_args(argv)


def convert(argv: list[str] | None = None) -> None:
    args = _parse_args(argv or sys.argv[1:])
    duckdb = _import_duckdb()
    conn = duckdb.connect(database=args.database)
    if args.threads:
        conn.execute(f"PRAGMA threads={int(args.threads)};")
    conn.execute("PRAGMA enable_progress_bar=false;")

    ts_col = args.timestamp_column
    date_col = args.date_column
    input_glob = args.input
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    query = f"""
    COPY (
        SELECT
            *,
            CAST({ts_col} AS TIMESTAMP) AS {ts_col}_ts,
            CAST(date_trunc('day', CAST({ts_col} AS TIMESTAMP)) AS DATE) AS {date_col}
        FROM read_ndjson_auto('{input_glob}')
    )
    TO '{output_dir.as_posix()}'
    (FORMAT PARQUET, COMPRESSION {args.compression}, PARTITION_BY ({date_col}));
    """
    conn.execute(query)
    conn.close()


if __name__ == "__main__":  # pragma: no cover
    convert()
