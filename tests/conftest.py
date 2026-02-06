from __future__ import annotations

import sys
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for candidate in [cur, *cur.parents]:
        if (candidate / "eqnet").is_dir() and (candidate / "tests").is_dir():
            return candidate
    return cur


repo_root = _find_repo_root(Path(__file__).parent)
repo_root_str = str(repo_root)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

