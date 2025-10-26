"""Restore config/ and persona.yaml from a snapshot directory."""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional
from uuid import uuid4


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EQNet config rollback utility")
    parser.add_argument("snapshot", type=Path, help="復元するスナップショット（ディレクトリ名またはパス）")
    parser.add_argument("--snapshots", type=Path, default=Path("snapshots"), help="スナップショット保存先ルート")
    parser.add_argument("--config-dir", type=Path, default=Path("config"), help="復元先 config ディレクトリ")
    parser.add_argument("--persona", type=Path, default=Path("persona.yaml"), help="復元先 persona ファイル")
    parser.add_argument("--dry-run", action="store_true", help="実際には変更せず、手順のみ表示する")
    return parser.parse_args(argv)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M")


def resolve_snapshot(project_root: Path, snapshots_root: Path, snapshot: Path) -> Path:
    candidate = snapshot
    if not candidate.is_absolute():
        candidate = (snapshots_root / snapshot).resolve()
    if not candidate.exists():
        raise SystemExit(f"スナップショットが見つかりません: {snapshot}")
    if not candidate.is_dir():
        raise SystemExit(f"スナップショットパスがディレクトリではありません: {candidate}")
    return candidate


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    snapshots_root = (project_root / args.snapshots).resolve()
    config_dir = (project_root / args.config_dir).resolve()
    persona_path = (project_root / args.persona).resolve()
    snapshot_dir = resolve_snapshot(project_root, snapshots_root, args.snapshot)
    snapshot_config = snapshot_dir / "config"
    if not snapshot_config.exists():
        raise SystemExit(f"config ディレクトリがスナップショットに存在しません: {snapshot_config}")

    snapshot_persona = snapshot_dir / "persona.yaml"
    operations = []

    tmp_restore = project_root / f".tmp_config_restore_{uuid4().hex}"
    backup_root = snapshots_root / "_pre_restore"
    backup_dir = backup_root / timestamp()

    operations.append(f"Copy snapshot config -> {tmp_restore}")
    operations.append(f"Backup current config -> {backup_dir / 'config'}")
    if snapshot_persona.exists():
        operations.append(f"Restore persona -> {persona_path}")

    if args.dry_run:
        print("Dry run. Planned operations:")
        for op in operations:
            print(f" - {op}")
        return 0

    if tmp_restore.exists():
        shutil.rmtree(tmp_restore)
    shutil.copytree(snapshot_config, tmp_restore)

    backup_root.mkdir(parents=True, exist_ok=True)
    if config_dir.exists():
        shutil.copytree(config_dir, backup_dir / "config")
        shutil.rmtree(config_dir)
    shutil.move(tmp_restore, config_dir)

    if snapshot_persona.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        if persona_path.exists():
            shutil.copy2(persona_path, backup_dir / "persona.yaml")
        shutil.copy2(snapshot_persona, persona_path)

    (backup_dir / "manifest.txt").write_text(
        f"Restored from: {snapshot_dir}\nCreated at: {datetime.now().isoformat(timespec='seconds')}\n",
        encoding="utf-8",
    )
    print(f"Restored configuration from {snapshot_dir}")
    print(f"Previous configuration backup: {backup_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
