from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_plugin_manifest_points_to_skills_directory() -> None:
    manifest_path = REPO_ROOT / ".codex-plugin" / "plugin.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert data["name"] == "eqnet-engineering-workflows"
    assert data["skills"] == "./skills/"


def test_expected_skill_directories_exist() -> None:
    expected = [
        "codebase-understanding",
        "frontend",
        "api",
        "qa-evals",
    ]

    for skill_name in expected:
        skill_path = REPO_ROOT / "skills" / skill_name / "SKILL.md"
        assert skill_path.exists(), skill_name


def test_skill_frontmatter_has_name_and_description() -> None:
    skill_path = REPO_ROOT / "skills" / "qa-evals" / "SKILL.md"
    text = skill_path.read_text(encoding="utf-8")

    assert text.startswith("---\n")
    assert "name: qa-evals" in text
    assert "description:" in text


def test_bundled_skill_resources_exist() -> None:
    expected_paths = [
        REPO_ROOT / "skills" / "codebase-understanding" / "templates" / "architecture-summary.md",
        REPO_ROOT / "skills" / "frontend" / "checklists" / "ui-review.md",
        REPO_ROOT / "skills" / "api" / "checklists" / "integration-upgrade.md",
        REPO_ROOT / "skills" / "qa-evals" / "checklists" / "regression-matrix.md",
    ]

    for path in expected_paths:
        assert path.exists(), str(path)
