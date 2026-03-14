from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "inner_os_standalone"
INNER_OS_SOURCE = REPO_ROOT / "inner_os"

PYPROJECT_TEXT = """[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "inner-os"
version = "0.1.0"
description = "Reusable inner-life operating system primitives"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
  "fastapi>=0.110",
  "numpy>=2.0",
]

[tool.setuptools]
packages = ["inner_os"]
"""

README_TEXT = """# inner_os standalone scaffold

This folder is generated from the emotional_dft workspace.

It contains a minimal exportable scaffold for `inner_os`:
- reusable inner-life cores
- service wrapper
- standalone HTTP app
- HTTP manifest

Run the standalone HTTP app with:

```bash
uvicorn inner_os.http_app:app --host 127.0.0.1 --port 8765
```
"""


def export_inner_os(output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_pkg = output_dir / "inner_os"
    shutil.copytree(INNER_OS_SOURCE, target_pkg)
    (output_dir / "pyproject.toml").write_text(PYPROJECT_TEXT, encoding="utf-8")
    (output_dir / "README.md").write_text(README_TEXT, encoding="utf-8")
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Export inner_os as a standalone scaffold")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    out = export_inner_os(args.output)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
