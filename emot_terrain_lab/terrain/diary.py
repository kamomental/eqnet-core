# -*- coding: utf-8 -*-
"""Daily diary generation for the emotional terrain system."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import numpy as np

from .llm import chat_text


@dataclass
class DiaryEntry:
    day: str
    text: str
    metrics: Dict[str, float]
    tags: List[str]
    highlights: List[str]
    culture_summary: Optional[Dict[str, Any]] = None

    def to_json(self) -> Dict:
        return {
            "day": self.day,
            "text": self.text,
            "metrics": self.metrics,
            "tags": self.tags,
            "highlights": self.highlights,
            "culture_summary": self.culture_summary,
        }

    @staticmethod
    def from_json(payload: Dict) -> "DiaryEntry":
        return DiaryEntry(
            day=payload["day"],
            text=payload.get("text", ""),
            metrics=dict(payload.get("metrics", {})),
            tags=list(payload.get("tags", [])),
            highlights=list(payload.get("highlights", [])),
            culture_summary=payload.get("culture_summary"),
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
        manager.entries = [
            DiaryEntry.from_json(obj) for obj in payload.get("entries", [])
        ]
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
        culture_stats: Optional[Dict[str, Dict[str, float]]] = None,
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
            culture_stats,
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
        culture_stats: Optional[Dict[str, Dict[str, float]]] = None,
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
        dominant_axes = ", ".join(top_axes[:3]) if top_axes else "縺ｪ縺・
        highlights = catalyst_highlights[:3]

        culture_summary, culture_lines = self._summarize_culture_stats(culture_stats)
        base_text = self._fallback_text(
            day_key,
            entropy,
            enthalpy,
            dissipation,
            dominant_axes,
            highlights,
            gentle_quotes,
            rest_snapshot,
            metrics,
            culture_lines=culture_lines,
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
                metrics,
                culture_lines=culture_lines,
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
            culture_summary=culture_summary,
        )
        return entry

    def _summarize_culture_stats(
        self, culture_stats: Optional[Dict[str, Dict[str, float]]]
    ) -> tuple[Optional[Dict[str, Any]], List[str]]:
        if not culture_stats:
            return None, []
        candidates: List[tuple[float, float, str, Dict[str, float]]] = []
        for tag, stats in culture_stats.items():
            try:
                count = float(stats.get("count", 0.0) or 0.0)
            except Exception:
                continue
            if count < 3.0:
                continue
            valence = stats.get("mean_valence")
            if valence is None:
                continue
            rho = stats.get("mean_rho")
            salience = abs(float(valence))
            if rho is not None:
                salience += 0.2 * abs(float(rho) - 0.5)
            candidates.append((salience, count, tag, stats))
        if not candidates:
            return None, []
        candidates.sort(key=lambda item: (-item[1], -item[0]))
        selected = candidates[:2]
        summary_tag = selected[0][2]
        summary = {"tag": summary_tag, "stats": selected[0][3]}
        lines: List[str] = []
        for _, _, tag, stats in selected:
            line = self._culture_line(tag, stats)
            if line:
                lines.append(line)
        return summary, lines

    def _culture_line(self, tag: str, stats: Dict[str, float]) -> Optional[str]:
        try:
            val = float(stats.get("mean_valence", 0.0) or 0.0)
        except Exception:
            val = 0.0
        rho_raw = stats.get("mean_rho")
        try:
            rho = float(rho_raw) if rho_raw is not None else 0.5
        except Exception:
            rho = 0.5
        if val <= -0.3:
            return f"{tag} の場面では、少し気持ちが沈みやすい時間が多かったみたい。"
        if val >= 0.3:
            return f"{tag} の場面では、あたたかさを感じる時間が多い日だったよ。"
        if rho >= 0.7:
            return f"{tag} の場面では、集中や一体感がぐっと高まる瞬間が印象的。"
        if rho <= 0.35:
            return f"{tag} の場面では、まだ距離を図りながら過ごす時間が多かったかも。"
        return None

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
        moment_metrics: Dict[str, float],
        culture_lines: Optional[List[str]] = None,
    ) -> str:
        lines = [
            f"縲須day_key} 縺ｮ繝弱・繝医・,
            f"繧ｨ繝ｳ繝医Ο繝斐・ {entropy:.2f} / 繧ｨ繝ｳ繧ｿ繝ｫ繝斐・ {enthalpy:.2f} / 謨｣騾ｸ {dissipation:.3f}縲・,
            f"縺阪ｇ縺・魂雎｡縺ｫ谿九▲縺溯ｻｸ・嘴dominant_axes}",
        ]
        if highlights:
            lines.append("隗ｦ蟐偵う繝吶Φ繝医・繝｡繝｢: " + " / ".join(highlights))
        if gentle_quotes:
            lines.append("莨夊ｩｱ繧ｹ繧ｱ繝・メ: " + " / ".join(gentle_quotes[:2]))
        if rest_snapshot.get("active"):
            until = rest_snapshot.get("rest_mode_until") or rest_snapshot.get("until")
            lines.append(f"繝ｬ繧ｹ繝医Δ繝ｼ繝峨ｒ邯ｭ謖∽ｸｭ縲よｬ｡縺ｮ蜀埼幕莠亥ｮ・ {until}")
        breath_line = self._format_breath_line(moment_metrics)
        if breath_line:
            lines.append(breath_line)
        heart_line = self._format_heart_line(moment_metrics)
        if heart_line:
            lines.append(heart_line)
        mask_line = self._format_mask_line(moment_metrics)
        if mask_line:
            lines.append(mask_line)
        if culture_lines:
            lines.extend(culture_lines)
        lines.append(
            "莉頑律繧ゆｸ邱偵↓險倬鹸繧帝ｲ繧√ｉ繧後∪縺励◆縲る㍾縺溘￠繧後・驕諷ｮ縺ｪ縺上せ繧ｭ繝・・縺励※縺上□縺輔＞縺ｭ縲・
        )
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
        moment_metrics: Dict[str, float],
        culture_lines: Optional[List[str]] = None,
    ) -> tuple[str, str]:
        system_prompt = (
            "縺ゅ↑縺溘・蜈ｱ逕溘ｒ蠢怜髄縺吶ｋ蠑ｱ縺БI縺ｧ縺吶・
            "莠ｺ髢薙↓驥崎差繧剃ｸ弱∴縺壹∫ｴ逶ｴ縺ｪ豌励▼縺阪ｒ繧・＆縺励＞譌･譛ｬ隱槭〒譌･險倥↓縺ｾ縺ｨ繧√∪縺吶・
            "隕ｳ蟇溘＆繧後◆謨ｰ蛟､繧剃ｺ句ｮ溘→縺励※霑ｰ縺ｹ縲∵─諠・・謗ｨ貂ｬ縺ｨ縺励※陦ｨ迴ｾ縺励※縺上□縺輔＞縲・
            "譌･險倥・譛蠕後↓縺ｯ縲∬ｪｭ繧莠ｺ縺檎┌逅・ｒ縺励↑縺上※繧医＞縺ｨ縺・≧荳譁・ｒ豺ｻ縺医※縺上□縺輔＞縲・
        )
        bullets = []
        bullets.append(f"- 譌･莉・ {day_key}")
        bullets.append(
            f"- 辭ｱ謖・ｨ・ 繧ｨ繝ｳ繝医Ο繝斐・ {entropy:.2f}, 繧ｨ繝ｳ繧ｿ繝ｫ繝斐・ {enthalpy:.2f}, 謨｣騾ｸ {dissipation:.3f}"
        )
        if top_axes:
            bullets.append(f"- 蠑ｷ縺剰｡ｨ繧後◆霆ｸ: {', '.join(top_axes[:4])}")
        if highlights:
            bullets.append(f"- 隗ｦ蟐偵・繝上う繝ｩ繧､繝・ {' / '.join(highlights[:3])}")
        if gentle_quotes:
            bullets.append(f"- 莨夊ｩｱ繧ｹ繧ｱ繝・メ: {' / '.join(gentle_quotes[:2])}")
        if rest_snapshot.get("active"):
            until = rest_snapshot.get("rest_mode_until") or rest_snapshot.get("until")
            bullets.append(f"- 繝ｬ繧ｹ繝医Δ繝ｼ繝臥ｶ咏ｶ壻ｸｭ (莠亥ｮ・ {until})")
        if loop_alert:
            bullets.append("- StoryGraph縺九ｉ繝ｫ繝ｼ繝怜だ蜷代・隴ｦ蜻翫≠繧奇ｼ磯撕縺九↓隕句ｮ医ｊ・・)
        if fatigue_flag:
            bullets.append("- 逍ｲ蜉ｴ邂｡逅・ヵ繝ｩ繧ｰ縺梧怏蜉ｹ")
        breath_line = self._format_breath_line(moment_metrics)
        if breath_line:
            bullets.append(f"- {breath_line}")
        heart_line = self._format_heart_line(moment_metrics)
        if heart_line:
            bullets.append(f"- {heart_line}")
        mask_line = self._format_mask_line(moment_metrics)
        if mask_line:
            bullets.append(f"- {mask_line}")
        if culture_lines:
            for line in culture_lines:
                bullets.append(f"- {line}")
        user_prompt = "莉頑律縺ｮ蜃ｺ譚･莠九ｒ蜆ｪ縺励￥譌･險倥↓縺励※縺上□縺輔＞:\n" + "\n".join(bullets)
        return system_prompt, user_prompt

    def _format_breath_line(self, metrics: Dict[str, float]) -> Optional[str]:
        total = float(metrics.get("fast_ack_samples") or 0.0)
        if total <= 0.0:
            return None
        parts: List[str] = []

        def _fmt(label: str, value: Optional[float]) -> Optional[str]:
            if value is None:
                return None
            pct = max(0.0, min(float(value), 1.0)) * 100.0
            return f"{label} {pct:.0f}%"

        for label, key in (
            ("縺・ｓ縺・ｓ", "ack_ratio"),
            ("繝悶Ξ繧ｹ", "breath_ratio"),
            ("髱吶°縺ｪ髢・, "silence_ratio"),
        ):
            snippet = _fmt(label, metrics.get(key))
            if snippet:
                parts.append(snippet)
        if not parts:
            return None
        return "蜻ｼ蜷ｸ繝ｭ繧ｰ: " + " / ".join(parts)

    def _format_heart_line(self, metrics: Dict[str, float]) -> Optional[str]:
        avg_rate = metrics.get("avg_heart_rate")
        if not avg_rate or avg_rate <= 0.0:
            return None
        bpm = float(avg_rate) * 60.0
        if bpm >= 90.0:
            tone = "蠢・牛縺後ｄ繧・ｫ倥ａ"
        elif bpm <= 55.0:
            tone = "縺ｨ縺ｦ繧る撕縺九↑鮠灘虚"
        else:
            tone = "遨上ｄ縺九↑鮠灘虚"
        return f"蠢・牛繝ｭ繧ｰ: {tone}・亥ｹｳ蝮・{bpm:.0f} bpm・・

    def _format_mask_line(self, metrics: Dict[str, float]) -> Optional[str]:
        mask = metrics.get("avg_masking_strength")
        if mask is None:
            return None

        value = float(mask)

        if value >= 0.6:
            # 譛ｬ髻ｳ繧偵°縺ｪ繧翫＠縺ｾ縺・％繧薙〒縺・◆
            tone = "譛ｬ蠖薙・豌玲戟縺｡繧偵＄縺｣縺ｨ縺励∪縺｣縺ｦ縲√′繧薙・縺｣縺ｦ縺・◆譎る俣縺悟､壹°縺｣縺滓律"
        elif value <= 0.3:
            # 縺九↑繧顔ｴ逶ｴ縺ｫ隧ｱ縺帙※縺・◆
            tone = "譛ｬ蠖薙・豌玲戟縺｡繧偵√ｏ繧翫→縺昴・縺ｾ縺ｾ蜃ｺ縺帙※縺・◆譌･"
        else:
            # 縺昴・荳ｭ髢・
            tone = "蟆代＠縺縺第悽髻ｳ繧偵＠縺ｾ縺・↑縺後ｉ繧ゅ√→縺薙ｍ縺ｩ縺薙ｍ邏逶ｴ縺ｫ隧ｱ縺帙※縺・◆譌･"

        # 謨ｰ蛟､繧よｮ九＠縺溘＞縺ｪ繧峨◎縺ｮ縺ｾ縺ｾ蜃ｺ縺・
        return f"縺薙％繧阪Γ繝｢: {tone}・医′縺ｾ繧薙Ξ繝吶Ν {value:.2f}・・

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

