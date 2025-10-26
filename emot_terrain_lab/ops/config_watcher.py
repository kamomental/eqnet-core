"""Config watcher / hot-reload coordinator for EQNet bus.

- Ctrl+R または SIGHUP で即時リロードを要求
- ファイル更新をポーリングし、変更を検出したら YAML 構文チェック
- 成功/失敗に関わらず {"type":"event","name":"config_reload"} を bus に publish
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

import yaml
from websockets import connect


class ConfigWatcher:
    def __init__(
        self,
        patterns: Sequence[str],
        *,
        root: Path,
        interval: float,
        bus_url: str,
    ) -> None:
        self.patterns = patterns
        self.root = root
        self.interval = interval
        self.bus_url = bus_url.rstrip("/")
        self.events_url = f"{self.bus_url}/events"
        self.logger = logging.getLogger("eqnet.config_watcher")
        self.mtimes: Dict[Path, float] = {}
        self.manual_trigger: asyncio.Event = asyncio.Event()
        self.stop_event: asyncio.Event = asyncio.Event()

    # -------------------------------------------------------------- path helper
    def resolve_files(self) -> Set[Path]:
        files: Set[Path] = set()
        for pattern in self.patterns:
            for path in self.root.glob(pattern):
                if path.is_file():
                    files.add(path)
        return files

    def refresh_mtimes(self, files: Iterable[Path]) -> None:
        for path in files:
            try:
                self.mtimes[path] = path.stat().st_mtime
            except FileNotFoundError:
                self.mtimes.pop(path, None)

    def scan_changes(self) -> List[Path]:
        changed: List[Path] = []
        current_files = self.resolve_files()
        # Detect new files
        for path in current_files:
            try:
                mtime = path.stat().st_mtime
            except FileNotFoundError:
                continue
            previous = self.mtimes.get(path)
            if previous is None or mtime > previous:
                changed.append(path)
                self.mtimes[path] = mtime

        # Detect deletions
        tracked = list(self.mtimes.keys())
        for path in tracked:
            if path not in current_files:
                changed.append(path)
                self.mtimes.pop(path, None)
        return sorted(set(changed))

    # --------------------------------------------------------------- bus helper
    async def publish(self, payload: Dict) -> None:
        data = json.dumps(payload, ensure_ascii=False)
        try:
            async with connect(self.events_url) as ws:
                await ws.send(data)
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Failed to publish config_reload event: %s", exc)

    async def reload_and_notify(self, changed: Sequence[Path], *, trigger: str) -> None:
        files = sorted(self.resolve_files())
        if not files:
            self.logger.warning("No config files matched: %s", ", ".join(self.patterns))
        errors: List[Dict[str, str]] = []
        for path in files:
            try:
                yaml.safe_load(path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                errors.append({"path": str(path), "error": str(exc)})

        level = "error" if errors else "info"
        notes = "manual_trigger" if trigger == "manual" else "fs_watch"
        payload = {
            "type": "event",
            "name": "config_reload",
            "level": level,
            "trigger": trigger,
            "notes": notes,
            "changed": [str(p) for p in changed] if changed else [str(p) for p in files],
        }
        if errors:
            payload["errors"] = errors
        await self.publish(payload)

    # --------------------------------------------------------------- main loops
    async def run(self) -> None:
        self.refresh_mtimes(self.resolve_files())
        loop = asyncio.get_running_loop()
        keyboard_stop = self._start_keyboard_listener(loop)
        try:
            while not self.stop_event.is_set():
                if self.manual_trigger.is_set():
                    self.manual_trigger.clear()
                    await self.reload_and_notify(self.resolve_files(), trigger="manual")
                    continue

                changed = self.scan_changes()
                if changed:
                    await self.reload_and_notify(changed, trigger="fs")
                await asyncio.sleep(self.interval)
        finally:
            keyboard_stop()

    def request_manual_reload(self) -> None:
        self.manual_trigger.set()

    def close(self) -> None:
        self.stop_event.set()

    # -------------------------------------------------------- keyboard listener
    def _start_keyboard_listener(self, loop: asyncio.AbstractEventLoop):
        stop_flag = threading.Event()

        def emit_reload() -> None:
            loop.call_soon_threadsafe(self.manual_trigger.set)

        if os.name == "nt":

            def run_windows() -> None:
                import msvcrt

                while not stop_flag.is_set():
                    if msvcrt.kbhit():
                        ch = msvcrt.getwch()
                        if ch == "\x12":  # Ctrl+R
                            emit_reload()
                    time.sleep(0.05)

            thread = threading.Thread(target=run_windows, name="config-watcher-hotkey", daemon=True)
            thread.start()

            def stop() -> None:
                stop_flag.set()
                thread.join(timeout=1.0)

            return stop

        def run_posix() -> None:
            import select
            import termios
            import tty

            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            try:
                while not stop_flag.is_set():
                    ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if ready:
                        ch = sys.stdin.read(1)
                        if ch == "\x12":  # Ctrl+R
                            emit_reload()
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        thread = threading.Thread(target=run_posix, name="config-watcher-hotkey", daemon=True)
        thread.start()

        def stop() -> None:
            stop_flag.set()
            thread.join(timeout=1.0)

        return stop


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EQNet config hot-reload watcher")
    parser.add_argument("--glob", action="append", default=["config/*.yaml"], help="監視する glob パターン (複数可)")
    parser.add_argument("--interval", type=float, default=1.0, help="変更ポーリング間隔（秒）")
    parser.add_argument("--bus-url", type=str, default="ws://127.0.0.1:8765", help="EQNet bus のベース URL（例: ws://127.0.0.1:8765）")
    parser.add_argument("--log-level", type=str, default="INFO", help="ロギングレベル")
    return parser.parse_args(argv)


async def async_main(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(levelname)s] %(asctime)s %(name)s: %(message)s",
    )
    watcher = ConfigWatcher(
        patterns=args.glob,
        root=Path("."),
        interval=args.interval,
        bus_url=args.bus_url,
    )

    loop = asyncio.get_running_loop()

    def handle_signal(signame: str) -> None:
        watcher.logger.info("Received %s, scheduling reload.", signame)
        watcher.request_manual_reload()

    for signame in ("SIGHUP",):
        if hasattr(signal, signame):
            try:
                loop.add_signal_handler(getattr(signal, signame), handle_signal, signame)
            except NotImplementedError:
                signal.signal(getattr(signal, signame), lambda _s, _f: handle_signal(signame))  # type: ignore[arg-type]

    for signame in ("SIGINT", "SIGTERM"):
        if hasattr(signal, signame):
            try:
                loop.add_signal_handler(getattr(signal, signame), watcher.close)
            except NotImplementedError:
                signal.signal(getattr(signal, signame), lambda _s, _f: watcher.close())  # type: ignore[arg-type]

    await watcher.run()
    return 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
