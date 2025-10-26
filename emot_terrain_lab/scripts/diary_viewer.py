# -*- coding: utf-8 -*-
"""
Interactive diary viewer using textual.

Usage:
    python scripts/diary_viewer.py --state data/state
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.reactive import reactive
from textual.widgets import Static, DataTable


def load_diary(state_dir: Path) -> list[dict]:
    payload = json.loads((state_dir / "diary.json").read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    entries.sort(key=lambda item: item.get("day", ""), reverse=True)
    return entries


class DiaryTable(DataTable):
    def __init__(self, entries: list[dict]) -> None:
        super().__init__(zebra_stripes=True)
        self.entries = entries

    def on_mount(self) -> None:
        self.add_columns("Day", "Entropy", "Enthalpy", "Tags")
        for entry in self.entries:
            metrics = entry.get("metrics", {})
            self.add_row(
                entry.get("day", ""),
                f"{metrics.get('entropy', 0):.2f}",
                f"{metrics.get('enthalpy', 0):.4f}",
                ", ".join(entry.get("tags", [])),
            )
        if self.rows:
            self.cursor_type = "row"
            self.focus()


class DiaryViewerApp(App):
    CSS = """
    Screen {
        padding: 1 2;
    }
    #body {
        border: tall $accent;
        padding: 1;
    }
    """

    selected_entry = reactive(dict)

    def __init__(self, entries: list[dict]) -> None:
        super().__init__()
        self.entries = entries
        self.table = DiaryTable(entries)
        self.text = Static("", id="body")

    def compose(self) -> ComposeResult:
        yield Container(self.table, self.text)

    def on_mount(self) -> None:
        if self.entries:
            self._show_entry(0)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self._show_entry(event.row_index)

    def _show_entry(self, index: int) -> None:
        entry = self.entries[index]
        metrics = entry.get("metrics", {})
        lines = [
            f"[b]Day[/b]: {entry.get('day', '')}",
            f"[b]Entropy[/b]: {metrics.get('entropy', 0):.4f}",
            f"[b]Enthalpy[/b]: {metrics.get('enthalpy', 0):.4f}",
            f"[b]Info flux[/b]: {metrics.get('info_flux', 0):.4f}",
            f"[b]Tags[/b]: {', '.join(entry.get('tags', []))}",
            "",
            entry.get("text", "").strip(),
        ]
        self.text.update("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=str, required=True, help="State directory (e.g., data/state)")
    args = parser.parse_args()
    entries = load_diary(Path(args.state))
    if not entries:
        print("No diary entries found.")
        return
    app = DiaryViewerApp(entries)
    app.run()


if __name__ == "__main__":
    main()
