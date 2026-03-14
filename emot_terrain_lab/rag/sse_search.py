from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


LOGGER = logging.getLogger(__name__)


def _env_flag(name: str, default: str = "0") -> bool:
    return (os.getenv(name, default) or default).strip().lower() in {"1", "true", "on", "yes"}


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            yield payload


def _row_id(row: Dict[str, Any], fallback_index: int) -> str:
    for key in ("id", "node_id", "turn_id", "event_id", "trace_id", "uuid"):
        value = row.get(key)
        if value:
            return str(value)
    return f"row-{fallback_index}"


def _row_text(row: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key in (
        "text",
        "content",
        "summary",
        "message",
        "user_text",
        "assistant_text",
        "memory_text",
        "label",
        "title",
    ):
        text = _safe_text(row.get(key))
        if text:
            parts.append(text)
    meta = row.get("meta")
    if isinstance(meta, dict):
        for key in ("title", "summary", "topic", "place", "person"):
            text = _safe_text(meta.get(key))
            if text:
                parts.append(text)
    return "\n".join(part for part in parts if part).strip()


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _normalize(vec: Sequence[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if not math.isfinite(norm) or norm <= 1e-12:
        return arr
    return arr / norm


def _parse_extra_paths(raw: str) -> Tuple[Path, ...]:
    items = [item.strip() for item in (raw or "").replace(";", ",").split(",") if item.strip()]
    return tuple(Path(item) for item in items)


@dataclass
class SSESearchConfig:
    model_name: str = "RikkaBotan/stable-static-embedding-fast-retrieval-mrl-ja"
    local_dir: Path = Path("models/sse/stable-static-embedding-fast-retrieval-mrl-ja")
    auto_download: bool = True
    memory_jsonl_path: Path = Path("data/logs.jsonl")
    extra_memory_jsonl_paths: Tuple[Path, ...] = (Path("logs/vision_memory.jsonl"),)
    top_k: int = 4
    max_chars_per_item: int = 280
    trust_remote_code: bool = False
    enabled: bool = False

    @classmethod
    def from_env(cls) -> "SSESearchConfig":
        extra_raw = os.getenv("EQNET_SSE_EXTRA_MEMORY_JSONL", "logs/vision_memory.jsonl")
        return cls(
            model_name=os.getenv(
                "EQNET_SSE_MODEL",
                "RikkaBotan/stable-static-embedding-fast-retrieval-mrl-ja",
            ),
            local_dir=Path(
                os.getenv(
                    "EQNET_SSE_LOCAL_DIR",
                    "models/sse/stable-static-embedding-fast-retrieval-mrl-ja",
                )
            ),
            auto_download=_env_flag("EQNET_SSE_AUTO_DOWNLOAD", "1"),
            memory_jsonl_path=Path(os.getenv("EQNET_SSE_MEMORY_JSONL", "data/logs.jsonl")),
            extra_memory_jsonl_paths=_parse_extra_paths(extra_raw),
            top_k=int(os.getenv("EQNET_SSE_TOPK", "4")),
            max_chars_per_item=int(os.getenv("EQNET_SSE_MAX_CHARS", "280")),
            trust_remote_code=_env_flag("EQNET_SSE_TRUST_REMOTE_CODE", "0"),
            enabled=_env_flag("EQNET_SSE_ENABLED", "0"),
        )


@dataclass
class SSESearchHit:
    row_id: str
    text: str
    score: float
    source: str = "sse"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "row_id": self.row_id,
            "text": self.text,
            "score": self.score,
            "source": self.source,
        }


class SSESearchAdapter:
    def __init__(self, config: Optional[SSESearchConfig] = None) -> None:
        self.config = config or SSESearchConfig.from_env()
        self._model: Any = None
        self._rows: Optional[List[Dict[str, Any]]] = None
        self._matrix: Optional[np.ndarray] = None

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def _memory_paths(self) -> List[Path]:
        paths = [self.config.memory_jsonl_path]
        for path in self.config.extra_memory_jsonl_paths:
            if path not in paths:
                paths.append(path)
        return paths

    def _resolve_model_source(self) -> str:
        local_dir = self.config.local_dir
        if local_dir.exists():
            return str(local_dir)
        if not self.config.auto_download:
            raise RuntimeError(f"SSE local model directory not found: {local_dir}")
        local_dir.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info(
            "SSE local model not found at %s; downloading %s",
            local_dir,
            self.config.model_name,
        )
        return self.config.model_name

    def _model_source_label(self) -> str:
        if self.config.local_dir.exists():
            return str(self.config.local_dir)
        return self.config.model_name

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "sentence-transformers is required for EQNET_SSE_ENABLED=1"
            ) from exc
        self._model = SentenceTransformer(
            self._resolve_model_source(),
            trust_remote_code=self.config.trust_remote_code,
        )
        return self._model

    def _load_rows(self) -> List[Dict[str, Any]]:
        if self._rows is not None:
            return self._rows
        rows: List[Dict[str, Any]] = []
        for source_path in self._memory_paths():
            for index, row in enumerate(_iter_jsonl(source_path)):
                text = _row_text(row)
                if not text:
                    continue
                rows.append(
                    {
                        "row_id": _row_id(row, index),
                        "text": text,
                        "source_path": str(source_path),
                    }
                )
        self._rows = rows
        return rows

    def _build_matrix(self) -> None:
        if self._matrix is not None:
            return
        rows = self._load_rows()
        if not rows:
            self._matrix = np.zeros((0, 0), dtype=np.float32)
            return
        model = self._load_model()
        texts = [row["text"] for row in rows]
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        self._matrix = np.asarray(vectors, dtype=np.float32)

    def search(self, query: str, *, top_k: Optional[int] = None) -> List[SSESearchHit]:
        query = (query or "").strip()
        if not query or not self.enabled:
            return []
        self._build_matrix()
        rows = self._load_rows()
        if self._matrix is None or self._matrix.size == 0 or not rows:
            return []
        model = self._load_model()
        query_vec = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        query_arr = _normalize(query_vec)
        scores = np.dot(self._matrix, query_arr)
        rank_count = max(1, int(top_k or self.config.top_k))
        top_indices = np.argsort(scores)[::-1][:rank_count]
        hits: List[SSESearchHit] = []
        for idx in top_indices:
            row = rows[int(idx)]
            hit = SSESearchHit(
                row_id=str(row["row_id"]),
                text=_truncate(str(row["text"]), self.config.max_chars_per_item),
                score=float(scores[int(idx)]),
            )
            hit.source = str(row.get("source_path") or "sse")
            hits.append(hit)
        return hits

    def build_context(self, query: str, *, top_k: Optional[int] = None) -> str:
        hits = self.search(query, top_k=top_k)
        if not hits:
            return ""
        lines = ["[sse-recall]"]
        for hit in hits:
            lines.append(f"- ({hit.score:.3f}) {hit.text}")
        return "\n".join(lines)

    def summarize_hits(self, hits: Sequence[SSESearchHit]) -> Dict[str, Any]:
        return {
            "backend": "sse",
            "model": self.config.model_name,
            "model_source": self._model_source_label(),
            "memory_jsonl_paths": [str(path) for path in self._memory_paths()],
            "hit_count": len(hits),
            "hits": [hit.to_dict() for hit in hits],
        }
