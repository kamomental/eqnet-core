"""DeepSeek‑OCR 接続バックエンド（HTTP/CLI 両対応）。

使い方（HTTP）
- サーバが `POST /infer` に multipart/form-data {file} を受け取り、Markdown（text/markdown または text/plain）を返すと仮定。
  `deepseek_http_backend(path, endpoint)` を VisionToMarkdown に渡す。

使い方（CLI）
- ローカルの DeepSeek‑OCR CLI（例: `deepseek-ocr`）を `--input <file>` `--output -` で実行し、標準出力の Markdown を受け取る。
  `deepseek_cli_backend(path, bin_path)` を渡す。

依存
- HTTP 版は標準ライブラリのみで実装（`urllib`）。CLI 版は `subprocess` を使用。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Callable
import subprocess
import sys
import urllib.request
import urllib.error
import mimetypes
import uuid
import os


def deepseek_http_backend(path: Path, endpoint: str) -> str:
    if not endpoint:
        raise ValueError("endpoint is required for HTTP backend")
    data, content_type = _encode_multipart(path)
    req = urllib.request.Request(endpoint, data=data)
    req.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
            return raw.decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.reason}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Connection error: {e.reason}") from e


def deepseek_cli_backend(path: Path, bin_path: Optional[str] = None) -> str:
    cmd = [bin_path or "deepseek-ocr", "--input", str(path), "--output", "-"]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=120)
    except subprocess.CalledProcessError as e:
        msg = e.output.decode("utf-8", errors="replace")
        raise RuntimeError(f"CLI failed: {msg}") from e
    except FileNotFoundError as e:
        raise RuntimeError(f"CLI not found: {cmd[0]}") from e
    return out.decode("utf-8", errors="replace")


def _encode_multipart(path: Path):
    boundary = uuid.uuid4().hex
    ctype = f"multipart/form-data; boundary={boundary}"
    # file part
    filename = path.name
    guessed = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    with path.open("rb") as fh:
        file_bytes = fh.read()
    lines = []
    add = lines.append
    add(f"--{boundary}\r\n".encode("utf-8"))
    add(
        (
            f"Content-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
            f"Content-Type: {guessed}\r\n\r\n"
        ).encode("utf-8")
    )
    add(file_bytes)
    add(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(lines)
    return body, ctype


__all__ = ["deepseek_http_backend", "deepseek_cli_backend"]

# ------------------------------ convenience wrapper

MarkdownBackend = Callable[[Path], str]


def get_http_endpoint_from_env() -> str:
    """Return DeepSeek-OCR endpoint from env, else empty string.

    Uses DEEPSEEK_OCR_ENDPOINT, then FALLBACK_OCR_ENDPOINT for compatibility.
    """
    return os.getenv("DEEPSEEK_OCR_ENDPOINT") or os.getenv("FALLBACK_OCR_ENDPOINT", "")


def get_cli_bin_from_env() -> str:
    """Return DeepSeek-OCR CLI path from env, else empty string.

    Uses DEEPSEEK_OCR_BIN, then looks for `deepseek-ocr` on PATH at runtime.
    """
    return os.getenv("DEEPSEEK_OCR_BIN", "")


def select_backend(mode: str, *, endpoint: str = "", bin_path: str = "") -> MarkdownBackend:
    """Select a backend by mode, pulling defaults from environment when missing.

    mode: "deepseek-http" | "deepseek-cli" | "dummy"
    """
    mode = (mode or "dummy").lower()
    if mode == "deepseek-http":
        ep = endpoint or get_http_endpoint_from_env()
        if not ep:
            raise ValueError("DeepSeek-OCR HTTP endpoint is not set. Set --endpoint or DEEPSEEK_OCR_ENDPOINT")
        return lambda p: deepseek_http_backend(p, ep)
    if mode == "deepseek-cli":
        bin_resolved = bin_path or get_cli_bin_from_env() or "deepseek-ocr"
        return lambda p: deepseek_cli_backend(p, bin_resolved)
    # dummy fallback (no OCR available)
    from .vision_to_md import dummy_backend  # local import to avoid cycle

    return lambda p: dummy_backend(p)

