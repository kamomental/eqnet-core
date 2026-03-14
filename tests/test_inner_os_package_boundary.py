import ast
from pathlib import Path


INNER_OS_DIR = Path("inner_os")
FORBIDDEN_PREFIXES = (
    "emot_terrain_lab",
    "apps",
    "scripts",
)


def _import_targets(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    targets: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                targets.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.module is None:
                continue
            if node.module:
                targets.append(node.module)
    return targets


def test_inner_os_does_not_import_repo_specific_runtime_layers() -> None:
    offenders: list[tuple[str, str]] = []
    for path in sorted(INNER_OS_DIR.glob("*.py")):
        for target in _import_targets(path):
            if target.startswith(FORBIDDEN_PREFIXES):
                offenders.append((str(path), target))
    assert offenders == []
