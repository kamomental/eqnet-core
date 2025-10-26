"""Create timestamped snapshots of config/ and persona.yaml."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EQNet config snapshot utility")
    parser.add_argument("--dest", type=Path, default=Path("snapshots"), help="スナップショットを保存するディレクトリ")
    parser.add_argument("--label", type=str, default="", help="サフィックスラベル（例: nightly）")
    parser.add_argument("--config-dir", type=Path, default=Path("config"), help="コピー対象の config ディレクトリ")
    parser.add_argument("--persona", type=Path, default=Path("persona.yaml"), help="オプションの persona ファイル")
    parser.add_argument("--force", action="store_true", help="既存ディレクトリがあっても上書きする")
    return parser.parse_args(argv)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M")


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    config_dir = (project_root / args.config_dir).resolve()
    persona_path = (project_root / args.persona).resolve()
    dest_root = (project_root / args.dest).resolve()
    dest_root.mkdir(parents=True, exist_ok=True)

    if not config_dir.exists():
        raise SystemExit(f"config ディレクトリが見つかりません: {config_dir}")

    name = timestamp()
    if args.label:
        name = f"{name}-{args.label}"
    snapshot_dir = dest_root / name
    if snapshot_dir.exists():
        if not args.force:
            raise SystemExit(f"スナップショットが既に存在します: {snapshot_dir}")
        shutil.rmtree(snapshot_dir)
    snapshot_dir.mkdir()

    target_config = snapshot_dir / "config"
    shutil.copytree(config_dir, target_config)

    persona_copied = False
    if persona_path.exists():
        shutil.copy2(persona_path, snapshot_dir / "persona.yaml")
        persona_copied = True

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_config": str(config_dir),
        "persona": str(persona_path) if persona_copied else None,
        "files": sum(1 for _ in target_config.rglob("*") if _.is_file()),
    }
    (snapshot_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Snapshot created: {snapshot_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
