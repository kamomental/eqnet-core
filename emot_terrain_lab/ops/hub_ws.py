"""EQNet lightweight local message bus (WebSocket-based).

仕様:
- チャネル: /metrics, /controls, /events（config/bus.yaml で拡張可）
- 受信 JSON に ts が無い場合はミリ秒精度で自動追加
- metrics / controls は揮発キャッシュを持ち、JSONL で永続ログ
- containment / health フラグを監視し、即時フェイルセーフを配信
- GET /health で現在の状態を返却（websockets の process_request で実装）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from pathlib import Path
from types import FrameType
from typing import Any, Dict, Iterable, List, Optional, Set

import yaml
from websockets import serve
from websockets.server import WebSocketServerProtocol


@dataclass
class ChannelConfig:
    name: str
    cache: bool = True
    tail: Optional[str] = None


@dataclass
class DependencyConfig:
    name: str
    kind: str = "module"  # module | path
    value: str = ""
    optional: bool = False

    def check(self) -> bool:
        if self.kind == "module":
            import importlib.util

            return importlib.util.find_spec(self.value) is not None
        if self.kind == "path":
            return Path(self.value).exists()
        raise ValueError(f"Unsupported dependency kind: {self.kind}")


@dataclass
class SafetyConfig:
    containment_event: str = "containment_triggered"
    degraded_overrides: Dict[str, Any] = field(
        default_factory=lambda: {"webcam_enabled": False, "audio_capture": False}
    )
    broadcast_health: bool = True


@dataclass
class BusConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    channels: Dict[str, ChannelConfig] = field(default_factory=dict)
    log_dir: Path = Path("logs/bus")
    ensure_ascii: bool = False
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    dependencies: List[DependencyConfig] = field(default_factory=list)
    health_default: str = "ok"
    degraded_if_missing_dependency: bool = True


class MessageBus:
    def __init__(self, config: BusConfig) -> None:
        self.config = config
        self.log_dir = config.log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.state: Dict[str, Optional[Dict[str, Any]]] = {
            name: None for name, chan in config.channels.items() if chan.cache
        }
        self.subscribers: Dict[str, Set[WebSocketServerProtocol]] = {
            name: set() for name in config.channels
        }
        self.health_status: str = config.health_default
        self.stop_event: asyncio.Event = asyncio.Event()
        self.logger = logging.getLogger("eqnet.bus")
        self.logger.debug("MessageBus initialised with channels: %s", list(config.channels))

    # ------------------------------------------------------------------ utils
    @staticmethod
    def now_ms() -> int:
        return int(time.time() * 1000)

    def tail_path(self, channel: str) -> Optional[Path]:
        chan_cfg = self.config.channels.get(channel)
        if chan_cfg and chan_cfg.tail:
            return self.log_dir / chan_cfg.tail
        return None

    def log_path(self, channel: str) -> Path:
        return self.log_dir / f"{channel}.jsonl"

    # ----------------------------------------------------------------- logging
    async def persist(self, channel: str, payload: Dict[str, Any]) -> None:
        message = json.dumps(payload, ensure_ascii=self.config.ensure_ascii)
        path = self.log_path(channel)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.open("a", encoding="utf-8").write(message + "\n")
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Failed to append log for %s: %s", channel, exc)

        tail = self.tail_path(channel)
        if tail:
            try:
                tail.write_text(message, encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                self.logger.exception("Failed to update tail for %s: %s", channel, exc)

    # ---------------------------------------------------------------- broadcast
    async def broadcast(
        self,
        channel: str,
        payload: Dict[str, Any],
        *,
        persist: bool = True,
        update_cache: bool = True,
        origin: Optional[WebSocketServerProtocol] = None,
    ) -> None:
        if "ts" not in payload:
            payload["ts"] = self.now_ms()
        if persist:
            await self.persist(channel, payload)
        if update_cache and channel in self.state:
            self.state[channel] = payload

        if self.config.safety.broadcast_health and channel == "controls":
            payload.setdefault("health", self.health_status)

        message = json.dumps(payload, ensure_ascii=self.config.ensure_ascii)
        dead: List[WebSocketServerProtocol] = []
        for ws in list(self.subscribers.get(channel, set())):
            if not ws.open:
                dead.append(ws)
                continue
            try:
                await ws.send(message)
            except Exception:  # noqa: BLE001
                self.logger.warning("Dropping subscriber on %s due to send failure.", channel)
                dead.append(ws)

        for ws in dead:
            self.subscribers[channel].discard(ws)

    # --------------------------------------------------------------- events API
    async def emit_event(self, name: str, level: str = "info", **extra: Any) -> None:
        payload: Dict[str, Any] = {"type": "event", "name": name, "level": level}
        payload.update(extra)
        await self.broadcast("events", payload, update_cache=False)

    # ------------------------------------------------------------- dependencies
    def check_dependencies(self) -> List[DependencyConfig]:
        missing: List[DependencyConfig] = []
        for dep in self.config.dependencies:
            try:
                ok = dep.check()
            except Exception as exc:  # noqa: BLE001
                self.logger.exception("Dependency check failed for %s: %s", dep.name, exc)
                ok = False
            if not ok:
                missing.append(dep)
                status = "optional" if dep.optional else "required"
                self.logger.warning("Dependency %s (%s) missing", dep.name, status)
        return missing

    async def handle_dependency_failures(self, missing: List[DependencyConfig]) -> None:
        if not missing:
            return
        names = [dep.name for dep in missing]
        await self.emit_event(
            "dependency_missing",
            level="warn",
            dependencies=names,
        )
        should_degrade = any(not dep.optional for dep in missing) and self.config.degraded_if_missing_dependency
        if should_degrade:
            self.logger.warning("Entering degraded mode due to missing dependencies: %s", names)
            await self.set_health("degraded", reason="missing_dependency", detail={"dependencies": names})

    # ---------------------------------------------------------------- health API
    async def set_health(self, status: str, *, reason: Optional[str] = None, detail: Optional[Dict[str, Any]] = None) -> None:
        if self.health_status == status:
            return
        self.health_status = status
        await self.emit_event("health_status", level="info", status=status, reason=reason, detail=detail or {})
        if status == "degraded":
            overrides = dict(self.config.safety.degraded_overrides)
            overrides.update({"type": "controls", "health": status})
            await self.broadcast("controls", overrides, update_cache=True)

    async def enforce_containment(self, source_payload: Dict[str, Any]) -> None:
        await self.emit_event(
            self.config.safety.containment_event,
            level="alert",
            notes="Containment switch engaged",
            source=source_payload.get("source", "bus"),
        )

    # -------------------------------------------------------------- ws handlers
    async def register(self, channel: str, websocket: WebSocketServerProtocol) -> None:
        self.subscribers.setdefault(channel, set()).add(websocket)
        cache = self.state.get(channel)
        if cache:
            try:
                await websocket.send(json.dumps(cache, ensure_ascii=self.config.ensure_ascii))
            except Exception:  # noqa: BLE001
                self.logger.warning("Failed to deliver cached state to new subscriber on %s", channel)

    async def unregister(self, channel: str, websocket: WebSocketServerProtocol) -> None:
        self.subscribers.get(channel, set()).discard(websocket)

    async def handle_message(self, channel: str, payload: Dict[str, Any]) -> None:
        if channel == "events":
            payload.setdefault("type", "event")
        else:
            payload.setdefault("type", channel)
        if "ts" not in payload:
            payload["ts"] = self.now_ms()

        if channel == "controls":
            if payload.get("containment"):
                await self.enforce_containment(payload)
            health = payload.get("health")
            if health and health != self.health_status:
                await self.set_health(str(health))
            if self.health_status == "degraded":
                for key, value in self.config.safety.degraded_overrides.items():
                    payload.setdefault(key, value)

        if channel == "events" and payload.get("type") == "ping":
            pong = {"type": "pong", "status": self.health_status}
            await self.broadcast("events", pong, update_cache=False, persist=False)
            return

        await self.broadcast(channel, payload, update_cache=True)

    async def connection_handler(self, websocket: WebSocketServerProtocol, path: str) -> None:
        channel = path.lstrip("/").split("?", 1)[0]
        if channel not in self.config.channels:
            self.logger.warning("Rejecting connection for unknown channel: %s", path)
            await websocket.close(code=4404, reason="unknown channel")
            return

        await self.register(channel, websocket)
        try:
            async for message in websocket:
                try:
                    payload = json.loads(message)
                except json.JSONDecodeError:
                    self.logger.warning("Discarding non-JSON message on %s: %s", channel, message)
                    continue
                await self.handle_message(channel, payload)
        finally:
            await self.unregister(channel, websocket)

    # ------------------------------------------------------------- HTTP health
    async def process_request(self, path: str, request_headers: Any) -> Optional[tuple[int, Iterable[tuple[str, str]], bytes]]:
        if path == "/health":
            body = json.dumps({"status": self.health_status, "ts": self.now_ms()}, ensure_ascii=self.config.ensure_ascii).encode("utf-8")
            headers = [("Content-Type", "application/json; charset=utf-8"), ("Content-Length", str(len(body)))]
            return HTTPStatus.OK, headers, body
        return None

    # ----------------------------------------------------------------- lifecycle
    async def start(self) -> None:
        missing = self.check_dependencies()
        await self.handle_dependency_failures(missing)

    def initiate_shutdown(self) -> None:
        self.stop_event.set()

    async def wait_for_shutdown(self) -> None:
        await self.stop_event.wait()


def load_config(path: Path, *, host_override: Optional[str], port_override: Optional[int]) -> BusConfig:
    if not path.exists():
        data: Dict[str, Any] = {}
    else:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    server = data.get("server", {})
    host = host_override or server.get("host", "127.0.0.1")
    port = port_override or int(server.get("port", 8765))

    raw_channels = data.get("channels") or {"metrics": {}, "controls": {}, "events": {}}
    channels: Dict[str, ChannelConfig] = {}
    if isinstance(raw_channels, dict):
        for name, cfg in raw_channels.items():
            channels[name] = ChannelConfig(
                name=name,
                cache=bool(cfg.get("cache", True)),
                tail=cfg.get("tail"),
            )
    elif isinstance(raw_channels, list):
        for name in raw_channels:
            channels[name] = ChannelConfig(name=name)
    else:
        raise ValueError("channels must be dict or list")

    logging_section = data.get("logging", {})
    log_dir = Path(logging_section.get("dir", "logs/bus"))
    ensure_ascii = bool(logging_section.get("ensure_ascii", False))

    safety_section = data.get("safety", {})
    safety = SafetyConfig(
        containment_event=safety_section.get("containment_event", "containment_triggered"),
        degraded_overrides=dict(safety_section.get("degraded_overrides") or {"webcam_enabled": False, "audio_capture": False}),
        broadcast_health=bool(safety_section.get("broadcast_health", True)),
    )

    dependencies_section = data.get("dependencies") or []
    dependencies: List[DependencyConfig] = []
    for dep in dependencies_section:
        if isinstance(dep, str):
            dependencies.append(DependencyConfig(name=dep, kind="module", value=dep))
            continue
        dependencies.append(
            DependencyConfig(
                name=str(dep.get("name")),
                kind=str(dep.get("kind", "module")),
                value=str(dep.get("value", dep.get("name"))),
                optional=bool(dep.get("optional", False)),
            )
        )

    health_section = data.get("health", {})
    health_default = str(health_section.get("default", "ok"))
    degraded_if_missing_dependency = bool(health_section.get("degraded_if_missing_dependency", True))

    return BusConfig(
        host=host,
        port=port,
        channels=channels,
        log_dir=log_dir,
        ensure_ascii=ensure_ascii,
        safety=safety,
        dependencies=dependencies,
        health_default=health_default,
        degraded_if_missing_dependency=degraded_if_missing_dependency,
    )


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EQNet lightweight WebSocket hub")
    parser.add_argument("--config", type=Path, default=Path("config/bus.yaml"), help="YAML 設定ファイルへのパス")
    parser.add_argument("--host", type=str, default=None, help="サーバーのバインド先（設定を上書き）")
    parser.add_argument("--port", type=int, default=None, help="ポート番号（設定を上書き）")
    parser.add_argument("--log-level", type=str, default="INFO", help="ロギングレベル")
    return parser.parse_args(argv)


async def async_main(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(levelname)s] %(asctime)s %(name)s: %(message)s",
    )
    config = load_config(args.config, host_override=args.host, port_override=args.port)
    bus = MessageBus(config)

    loop = asyncio.get_running_loop()

    def handle_signal(signame: str) -> None:
        bus.logger.info("Received %s, shutting down...", signame)
        bus.initiate_shutdown()

    for signame in ("SIGINT", "SIGTERM"):
        if hasattr(signal, signame):
            try:
                loop.add_signal_handler(getattr(signal, signame), handle_signal, signame)
            except NotImplementedError:
                signal.signal(getattr(signal, signame), lambda signum, frame: handle_signal(signal.Signals(signum).name))  # type: ignore[arg-type]

    async with serve(
        lambda websocket, path: bus.connection_handler(websocket, path),
        config.host,
        config.port,
        process_request=bus.process_request,
    ) as ws_server:
        bus.logger.info("EQNet bus listening on ws://%s:%s", config.host, config.port)
        await bus.start()
        await bus.wait_for_shutdown()
        bus.logger.info("Stopping bus...")
        ws_server.close()
        await ws_server.wait_closed()
    return 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"Bus crashed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
