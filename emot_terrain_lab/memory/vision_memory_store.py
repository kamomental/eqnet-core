from __future__ import annotations

from dataclasses import dataclass
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


@dataclass
class VisionMemoryStore:
    path: Path = Path("logs/vision_memory.jsonl")
    dedupe_window_seconds: float = 180.0
    dedupe_min_token_overlap: float = 0.72
    recent_scan_limit: int = 12

    def _entry_id(
        self,
        *,
        turn_id: int,
        timestamp: float,
        perception_summary: Optional[Dict[str, Any]],
    ) -> str:
        response_id = None
        if isinstance(perception_summary, dict):
            candidate = perception_summary.get("response_id")
            if isinstance(candidate, str) and candidate.strip():
                response_id = candidate.strip()
        ts_ms = int(float(timestamp) * 1000)
        if response_id:
            return f"vision-{ts_ms}-{response_id[-12:]}"
        return f"vision-{ts_ms}-{int(turn_id)}"

    def append_observed(
        self,
        *,
        perception_summary: Optional[Dict[str, Any]],
        turn_id: int,
        session_id: Optional[str],
        talk_mode: str,
        response_route: str,
        user_text: Optional[str] = None,
        image_path: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(perception_summary, dict):
            return None
        text_value = perception_summary.get("text")
        if not isinstance(text_value, str) or not text_value.strip():
            return None

        ts_value = float(timestamp if timestamp is not None else time.time())
        payload = {
            "id": self._entry_id(
                turn_id=turn_id,
                timestamp=ts_value,
                perception_summary=perception_summary,
            ),
            "schema": "observed_vision/v1",
            "kind": "observed",
            "modality": "vision",
            "turn_id": int(turn_id),
            "session_id": session_id,
            "timestamp": ts_value,
            "type": "vision_summary",
            "text": text_value.strip(),
            "summary": text_value.strip(),
            "user_text": (user_text or "").strip() or None,
            "image_path": image_path,
            "talk_mode": talk_mode,
            "response_route": response_route,
            "meta": {
                "model": perception_summary.get("model"),
                "backend": perception_summary.get("backend"),
                "response_id": perception_summary.get("response_id"),
            },
        }

        duplicate = self._find_duplicate(payload)
        if duplicate is not None:
            duplicate_meta = dict(duplicate.get("meta") or {})
            duplicate_meta["duplicate_suppressed"] = True
            duplicate_meta["duplicate_seen_at"] = ts_value
            duplicate_meta["duplicate_candidate_id"] = payload["id"]
            duplicate["meta"] = duplicate_meta
            duplicate["suppressed"] = True
            duplicate["suppressed_candidate_id"] = payload["id"]
            return duplicate

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return payload

    def _find_duplicate(self, candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        candidate_ts = float(candidate.get("timestamp") or 0.0)
        candidate_image = str(candidate.get("image_path") or "").strip()
        candidate_tokens = self._tokenize(candidate.get("summary") or candidate.get("text") or "")
        if not candidate_tokens:
            return None

        for record in self._iter_recent_records():
            if str(record.get("schema") or "") != "observed_vision/v1":
                continue
            record_ts = float(record.get("timestamp") or 0.0)
            if candidate_ts and record_ts and abs(candidate_ts - record_ts) > self.dedupe_window_seconds:
                continue
            record_image = str(record.get("image_path") or "").strip()
            if candidate_image and record_image and candidate_image != record_image:
                continue
            record_tokens = self._tokenize(record.get("summary") or record.get("text") or "")
            if not record_tokens:
                continue
            overlap = self._token_overlap(candidate_tokens, record_tokens)
            if overlap >= self.dedupe_min_token_overlap:
                return record
        return None

    def _iter_recent_records(self) -> Iterable[Dict[str, Any]]:
        if not self.path.exists():
            return []
        records = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return list(reversed(records[-self.recent_scan_limit :]))

    def _tokenize(self, text: str) -> set[str]:
        raw = str(text or "").strip().lower()
        if not raw:
            return set()
        pieces = [piece for piece in re.split(r"[^a-z0-9]+", raw) if len(piece) >= 3]
        if pieces:
            return set(pieces)
        compact = re.sub(r"\s+", "", raw)
        if not compact:
            return set()
        if len(compact) <= 3:
            return {compact}
        return {compact[idx : idx + 3] for idx in range(0, len(compact) - 2)}

    def _token_overlap(self, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        intersection = len(left & right)
        base = max(1, min(len(left), len(right)))
        return float(intersection) / float(base)
