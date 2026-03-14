from pathlib import Path

from scripts.export_inner_os_package import export_inner_os


def test_export_inner_os_creates_standalone_scaffold(tmp_path: Path) -> None:
    out = export_inner_os(tmp_path / "inner_os_pkg")
    assert (out / "pyproject.toml").exists()
    assert (out / "README.md").exists()
    assert (out / "inner_os" / "service.py").exists()
    assert (out / "inner_os" / "http_app.py").exists()
