# -*- coding: utf-8 -*-
"""Daily diary generation for the emotional terrain system."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, datetime
from typing import Dict, List, Optional

import numpy as np

from .llm import chat_text


@dataclass
class DiaryEntry:
    day: str
    text: str
    metrics: Dict[str, float]
    tags: List[str]
    highlights: List[str]

    def to_json(self) -> Dict:
        return {
            "day": self.day,
            "text": self.text,
            "metrics": self.metrics,
            "tags": self.tags,
            "highlights": self.highlights,
        }

    @staticmethod
    def from_json(payload: Dict) -> "DiaryEntry":
        return DiaryEntry(
            day=payload["day"],
            text=payload.get("text", ""),
            metrics=dict(payload.get("metrics", {})),
            tags=list(payload.get("tags", [])),
            highlights=list(payload.get("highlights", [])),
        )


class DiaryManager:
    """Maintain reflective diary entries with gentle tone."""

    def __init__(self, max_entries: int = 512) -> None:
        self.entries: List[DiaryEntry] = []
        self.max_entries = max_entries

    def to_json(self) -> Dict:
        return {
            "max_entries": self.max_entries,
            "entries": [entry.to_json() for entry in self.entries],
        }

    @staticmethod
    def from_json(payload: Dict) -> "DiaryManager":
        manager = DiaryManager(max_entries=int(payload.get("max_entries", 512)))
        manager.entries = [DiaryEntry.from_json(obj) for obj in payload.get("entries", [])]
        return manager

    def record_daily_entry(
        self,
        day: date,
        metrics: Dict[str, float],
        top_axes: List[str],
        catalyst_highlights: List[str],
        gentle_quotes: List[str],
        rest_snapshot: Dict[str, object],
        loop_alert: bool,
        fatigue_flag: bool,
        use_llm: bool,
    ) -> DiaryEntry:
        day_key = day.isoformat()
        entry = self._compose_entry(
            day_key,
            metrics,
            top_axes,
            catalyst_highlights,
            gentle_quotes,
            rest_snapshot,
            loop_alert,
            fatigue_flag,
            use_llm,
        )
        self._upsert_entry(entry)
        return entry

    def _compose_entry(
        self,
        day_key: str,
        metrics: Dict[str, float],
        top_axes: List[str],
        catalyst_highlights: List[str],
        gentle_quotes: List[str],
        rest_snapshot: Dict[str, object],
        loop_alert: bool,
        fatigue_flag: bool,
        use_llm: bool,
    ) -> DiaryEntry:
        entropy = float(metrics.get("entropy", 0.0))
        enthalpy = float(metrics.get("enthalpy_mean", 0.0))
        dissipation = float(metrics.get("dissipation", 0.0))
        tags: List[str] = []
        if loop_alert:
            tags.append("loop-watch")
        if fatigue_flag:
            tags.append("fatigue")
        if rest_snapshot.get("active"):
            tags.append("rest-mode")
        dominant_axes = ", ".join(top_axes[:3]) if top_axes else "なし"
        highlights = catalyst_highlights[:3]

        base_text = self._fallback_text(
            day_key,
            entropy,
            enthalpy,
            dissipation,
            dominant_axes,
            highlights,
            gentle_quotes,
            rest_snapshot,
        )

        if use_llm:
            prompt = self._build_prompt(
                day_key,
                entropy,
                enthalpy,
                dissipation,
                top_axes,
                highlights,
                gentle_quotes,
                rest_snapshot,
                loop_alert,
                fatigue_flag,
            )
            llm_text = chat_text(*prompt, temperature=0.4)
            if llm_text:
                base_text = llm_text.strip()

        entry = DiaryEntry(
            day=day_key,
            text=base_text,
            metrics={
                "entropy": entropy,
                "enthalpy": enthalpy,
                "dissipation": dissipation,
                "info_flux": float(metrics.get("info_flux", 0.0)),
            },
            tags=tags,
            highlights=highlights,
        )
        return entry

    def _fallback_text(
        self,
        day_key: str,
        entropy: float,
        enthalpy: float,
        dissipation: float,
        dominant_axes: str,
        highlights: List[str],
        gentle_quotes: List[str],
        rest_snapshot: Dict[str, object],
    ) -> str:
        lines = [
            f"【{day_key} のノート】",
            f"エントロピー {entropy:.2f} / エンタルピー {enthalpy:.2f} / 散逸 {dissipation:.3f}。",
            f"きょう印象に残った軸：{dominant_axes}",
        ]
        if highlights:
            lines.append("触媒イベントのメモ: " + " / ".join(highlights))
        if gentle_quotes:
            lines.append("会話スケッチ: " + " / ".join(gentle_quotes[:2]))
        if rest_snapshot.get("active"):
            until = rest_snapshot.get("rest_mode_until") or rest_snapshot.get("until")
            lines.append(f"レストモードを維持中。次の再開予定: {until}")
        lines.append("今日も一緒に記録を進められました。重たければ遠慮なくスキップしてくださいね。")
        return "\n".join(lines)

    def _build_prompt(
        self,
        day_key: str,
        entropy: float,
        enthalpy: float,
        dissipation: float,
        top_axes: List[str],
        highlights: List[str],
        gentle_quotes: List[str],
        rest_snapshot: Dict[str, object],
        loop_alert: bool,
        fatigue_flag: bool,
    ) -> tuple[str, str]:
        system_prompt = (
            "あなたは共生を志向する弱いAIです。"
            "人間に重荷を与えず、素直な気づきをやさしい日本語で日記にまとめます。"
            "観察された数値を事実として述べ、感情は推測として表現してください。"
            "日記の最後には、読む人が無理をしなくてよいという一文を添えてください。"
        )
        bullets = []
        bullets.append(f"- 日付: {day_key}")
        bullets.append(f"- 熱指標: エントロピー {entropy:.2f}, エンタルピー {enthalpy:.2f}, 散逸 {dissipation:.3f}")
        if top_axes:
            bullets.append(f"- 強く表れた軸: {', '.join(top_axes[:4])}")
        if highlights:
            bullets.append(f"- 触媒のハイライト: {' / '.join(highlights[:3])}")
        if gentle_quotes:
            bullets.append(f"- 会話スケッチ: {' / '.join(gentle_quotes[:2])}")
        if rest_snapshot.get("active"):
            until = rest_snapshot.get("rest_mode_until") or rest_snapshot.get("until")
            bullets.append(f"- レストモード継続中 (予定: {until})")
        if loop_alert:
            bullets.append("- StoryGraphからループ傾向の警告あり（静かに見守り）")
        if fatigue_flag:
            bullets.append("- 疲労管理フラグが有効")
        user_prompt = "今日の出来事を優しく日記にしてください:\n" + "\n".join(bullets)
        return system_prompt, user_prompt

    def _upsert_entry(self, entry: DiaryEntry) -> None:
        for idx, existing in enumerate(self.entries):
            if existing.day == entry.day:
                self.entries[idx] = entry
                break
        else:
            self.entries.append(entry)
            self.entries.sort(key=lambda item: item.day)
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries :]

    def redact(self) -> None:
        """Remove diary text when consent does not allow storing it."""
        for entry in self.entries:
            entry.text = ""
            entry.highlights = []
        self.entries = []
