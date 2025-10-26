"""Hotkey launcher for EQNet bus (F9/F10/F11 + Ctrl+C to quit)."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from typing import Iterable, Optional

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Static
from websockets import connect
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from websockets.legacy.client import WebSocketClientProtocol


class BusClient:
    def __init__(self, bus_url: str) -> None:
        self.bus_url = bus_url.rstrip("/")
        self.events_url = f"{self.bus_url}/events"
        self.ws: Optional[WebSocketClientProtocol] = None
        self.lock = asyncio.Lock()
        self.logger = logging.getLogger("eqnet.hotkeys.bus")

    async def ensure_connection(self) -> WebSocketClientProtocol:
        if self.ws and not self.ws.closed:
            return self.ws
        self.logger.debug("Opening WebSocket connection to %s", self.events_url)
        self.ws = await connect(self.events_url)
        return self.ws

    async def send_event(self, name: str, *, level: str = "info", **fields) -> None:
        payload = {"type": "event", "name": name, "level": level}
        payload.update(fields)
        data = json.dumps(payload, ensure_ascii=False)

        async with self.lock:
            attempts = 0
            while attempts < 2:
                attempts += 1
                try:
                    ws = await self.ensure_connection()
                    await ws.send(data)
                    return
                except (ConnectionClosedError, ConnectionClosedOK, OSError) as exc:
                    self.logger.warning("Lost connection to bus (%s), retrying...", exc)
                    if self.ws:
                        try:
                            await self.ws.close()
                        except Exception:  # noqa: BLE001
                            pass
                    self.ws = None
            raise RuntimeError("Failed to deliver hotkey event to bus")

    async def aclose(self) -> None:
        if self.ws and not self.ws.closed:
            await self.ws.close()
        self.ws = None


class HotkeyApp(App[None]):
    CSS = """
    Screen {
        layout: vertical;
        align-horizontal: center;
    }
    #status {
        padding: 1 2;
        border: wide $accent;
        margin: 1 2;
    }
    """

    BINDINGS = [
        Binding("f9", "toggle_session", "セッション開始/終了"),
        Binding("f10", "pause_audio", "音介入一時停止"),
        Binding("f11", "bookmark", "ログマーカー"),
        Binding("ctrl+c", "quit", "終了"),
    ]

    def __init__(self, bus_url: str) -> None:
        super().__init__()
        self.bus = BusClient(bus_url)
        self.status = Static(id="status")
        self.bus_url = bus_url

    def compose(self) -> ComposeResult:
        yield Static("EQNet Hotkeys\nF9: セッション切替 / F10: 音介入 pause / F11: ログマーカー", id="title")
        yield self.status
        yield Footer()

    async def on_mount(self) -> None:
        self.status.update("準備完了: bus に接続するホットキーを待機しています")

    async def action_toggle_session(self) -> None:
        await self._emit("session_toggle", notes="hotkey:F9")

    async def action_pause_audio(self) -> None:
        await self._emit("audio_pause_toggle", notes="hotkey:F10")

    async def action_bookmark(self) -> None:
        await self._emit("log_bookmark", level="info", notes="hotkey:F11")

    async def _emit(self, name: str, level: str = "info", **fields) -> None:
        try:
            await self.bus.send_event(name, level=level, **fields)
        except Exception as exc:  # noqa: BLE001
            self.status.update(f"[red]送信失敗[/red]: {exc}")
        else:
            self.status.update(f"[green]送信済み[/green]: {name}")

    async def on_unmount(self, event: events.Unmount) -> None:  # noqa: D401
        await self.bus.aclose()


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EQNet bus hotkey launcher")
    parser.add_argument("--bus-url", type=str, default="ws://127.0.0.1:8765", help="EQNet bus のベース URL")
    parser.add_argument("--log-level", type=str, default="INFO", help="ロギングレベル")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(levelname)s] %(asctime)s %(name)s: %(message)s",
    )
    app = HotkeyApp(args.bus_url)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
