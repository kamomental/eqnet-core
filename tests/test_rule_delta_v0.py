from __future__ import annotations

from pathlib import Path

from eqnet.runtime.rule_delta_v0 import RULE_DELTA_FILE, load_rule_deltas


def test_rule_delta_missing_or_empty_file_returns_noop(tmp_path: Path) -> None:
    assert load_rule_deltas(tmp_path) == []

    rule_path = tmp_path / RULE_DELTA_FILE
    rule_path.write_text("", encoding="utf-8")
    assert load_rule_deltas(tmp_path) == []
