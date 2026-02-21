# -*- coding: utf-8 -*-
"""
Minimal LLM hub wrapper.

Routes requests to the existing ``terrain.llm`` helpers while tracking metadata
so that the outer runtime can treat this as a structured "mouth". The routing
logic is intentionally lightweight; production deployments can swap it out with
tool-enabled planners without changing the call-site.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import time
import os

from terrain import llm as terrain_llm
from emot_terrain_lab.tools.registry import SkillRegistry
from emot_terrain_lab.hub.akorn import AkornGate, AkornConfig
from emot_terrain_lab.rag.lazy_rag import LazyRAG, LazyRAGConfig
from runtime.config import load_runtime_cfg


_LOGGER = logging.getLogger(__name__)


@dataclass
class LLMHubConfig:
    """Configuration for the hub interface."""

    default_system_prompt: str = (
        "You are an assistant voice for an EQNet-driven companion. "
        "Keep responses concise and compassionate."
    )
    default_intent: str = "chitchat"
    fallback_text: str = "ごめんね、今すぐ良い言葉が浮かばなかったよ。"


@dataclass
class HubResponse:
    text: str
    model: Optional[str]
    trace_id: str
    latency_ms: float
    controls_used: Dict[str, float]
    safety: Dict[str, str]
    confidence: float = 0.0
    uncertainty_reason: Tuple[str, ...] = ()


class LLMHub:
    """Very small routing facade over ``terrain.llm``."""

    def __init__(self, config: Optional[LLMHubConfig] = None) -> None:
        self.config = config or LLMHubConfig()
        self.registry = SkillRegistry()
        # Allow environment-based tuning of AKOrN small gains
        self.akorn = AkornGate(AkornConfig.from_env())
        self._lazy_rag: Optional[LazyRAG] = None
        try:
            self._runtime_cfg = load_runtime_cfg()
        except Exception:
            self._runtime_cfg = None

    def _get_lazy_rag(self) -> Optional[LazyRAG]:
        flag = (os.getenv("EQNET_LAZY_RAG", "0") or "0").lower()
        if flag in {"0", "false", "off"}:
            return None
        if self._lazy_rag is None:
            self._lazy_rag = LazyRAG(LazyRAGConfig.from_env())
        return self._lazy_rag

    def _show_uncertainty_meta(self) -> bool:
        env = (os.getenv("EQNET_SHOW_UNCERTAINTY_META", "") or "").strip().lower()
        if env in {"1", "true", "on"}:
            return True
        if env in {"0", "false", "off"}:
            return False
        ui_cfg = getattr(self._runtime_cfg, "ui", None)
        return bool(getattr(ui_cfg, "show_uncertainty_meta", True))

    def _ui_locale(self) -> str:
        env = (os.getenv("EQNET_UI_LOCALE", "") or "").strip()
        if env:
            return env
        return "ja"

    def _render_uncertainty_line(self, confidence: float, reasons: Tuple[str, ...]) -> str:
        lex = _uncertainty_lexicon(self._ui_locale())
        ui_cfg = getattr(self._runtime_cfg, "ui", None)
        if ui_cfg is not None:
            labels_cfg = getattr(ui_cfg, "uncertainty_reason_labels", None)
            if isinstance(labels_cfg, dict) and labels_cfg:
                merged_labels = dict(lex.get("reason_labels", {}))
                for code, label in labels_cfg.items():
                    if isinstance(code, str) and isinstance(label, str) and label.strip():
                        merged_labels[code] = label
                lex["reason_labels"] = merged_labels
            for key, attr in (
                ("line_template", "uncertainty_line_template"),
                ("reason_low", "uncertainty_reason_low"),
                ("reason_join", "uncertainty_reason_join"),
                ("confidence_label_low", "uncertainty_confidence_label_low"),
                ("confidence_label_mid", "uncertainty_confidence_label_mid"),
                ("confidence_label_high", "uncertainty_confidence_label_high"),
            ):
                value = getattr(ui_cfg, attr, None)
                if isinstance(value, str) and value.strip():
                    lex[key] = value
            for key, attr in (
                ("confidence_low_max", "uncertainty_confidence_low_max"),
                ("confidence_mid_max", "uncertainty_confidence_mid_max"),
            ):
                value = getattr(ui_cfg, attr, None)
                try:
                    lex[key] = float(value)
                except (TypeError, ValueError):
                    pass
        label_map = lex.get("reason_labels", {}) if isinstance(lex.get("reason_labels"), dict) else {}
        if reasons:
            translated = [str(label_map.get(code, code)) for code in reasons]
            reason_text = str(lex.get("reason_join", ", ")).join(translated)
        else:
            reason_text = str(lex.get("reason_low", "low"))
        confidence_label = _confidence_label(confidence, lex)
        template = str(lex.get("line_template", "confidence: {confidence:.2f} / reason: {reasons}"))
        return template.format(
            confidence=confidence,
            confidence_label=confidence_label,
            reasons=reason_text,
        )

    def _estimate_uncertainty(
        self,
        *,
        has_context: bool,
        sat_ratio: Optional[float],
        retrieval_error: bool,
    ) -> Tuple[float, Tuple[str, ...]]:
        confidence = 0.78
        reasons = []
        if not has_context:
            confidence -= 0.20
            reasons.append("retrieval_sparse")
        if retrieval_error:
            confidence -= 0.15
            reasons.append("retrieval_error")
        if sat_ratio is not None:
            if sat_ratio >= 0.80:
                confidence -= 0.25
                reasons.append("score_saturation_high")
            elif sat_ratio >= 0.60:
                confidence -= 0.12
                reasons.append("score_saturation_moderate")
        confidence = max(0.05, min(0.95, confidence))
        return confidence, tuple(reasons)

    def generate(
        self,
        user_text: str,
        context: Optional[str],
        controls: Dict[str, float],
        intent: Optional[str] = None,
        slos: Optional[Dict[str, float]] = None,
    ) -> HubResponse:
        """
        Generate text under EQNet control.

        Parameters
        ----------
        user_text:
            Raw user utterance.
        context:
            Optional additional context (e.g., retrieved snippets). For the
            scaffold we simply append it to the prompt.
        controls:
            Dict containing behaviour controls (pause_ms, temperature, warmth…)
            produced by :class:`hub.policy.PolicyHead`.
        intent:
            Intent label (qa/chitchat/code…). Currently only chitchat behaviour
            is defined but the parameter is kept for future routing policies.
        slos:
            Optional SLO hints (e.g., ``{"p95_ms": 180}``). The scaffold only
            records the value for metrics.
        """
        skill = self.registry.find_by_intent(intent or self.config.default_intent)
        system_prompt = self.config.default_system_prompt
        chosen_llm = None
        if skill:
            chosen_llm = skill.llm
            if skill.intent == "qa":
                system_prompt += " Use the provided context when helpful."
            if skill.context_sources and context:
                context = context.strip()
        retrieval_error = False
        sat_ratio: Optional[float] = None
        if not context:
            rag = self._get_lazy_rag()
            if rag is not None:
                try:
                    context = rag.build_context(user_text)
                except Exception:
                    retrieval_error = True
                    context = None
                diag = getattr(rag, "last_score_diag", {}) or {}
                try:
                    sat_ratio = float(diag.get("sat_ratio")) if "sat_ratio" in diag else None
                except (TypeError, ValueError):
                    sat_ratio = None
        prompt = user_text
        if context:
            prompt = context.strip() + "\n\n---\n\n" + user_text.strip()

        # Apply AKOrN gate (minimal): looks for R/rho/I/q in controls or nested under 'akorn'
        gated_controls, gate_log = self.akorn.apply(controls)
        temperature = float(gated_controls.get("temperature", controls.get("temperature", 0.6)))
        top_p = float(gated_controls.get("top_p", controls.get("top_p", 0.85)))
        pause_ms = float(gated_controls.get("pause_ms", controls.get("pause_ms", 360.0)))

        start = time.perf_counter()
        text = terrain_llm.chat_text(system_prompt, prompt, temperature=temperature, top_p=top_p)
        latency = (time.perf_counter() - start) * 1000.0

        if not text:
            text = self.config.fallback_text
        confidence, uncertainty_reason = self._estimate_uncertainty(
            has_context=bool(context),
            sat_ratio=sat_ratio,
            retrieval_error=retrieval_error,
        )
        if self._show_uncertainty_meta():
            text = f"{text}\n\n{self._render_uncertainty_line(confidence, uncertainty_reason)}"
        model = terrain_llm.get_llm().model or chosen_llm

        trace_id = f"hub-{int(time.time() * 1000)}"
        safety = {
            "rating": "G",
            "spoiler": controls.get("spoiler_mode", "warn"),
            "pii_block": str(True),
        }
        if slos and latency > slos.get("p95_ms", 1e9):
            safety["note"] = "latency_degraded"

        # Merge control usage for traceability
        used_controls = dict(gated_controls)
        used_controls.setdefault("temperature", temperature)
        used_controls.setdefault("top_p", top_p)
        used_controls.setdefault("pause_ms", pause_ms)

        # Attach AKOrN deltas into safety note
        if gate_log:
            gate_note = {k: float(v) for k, v in gate_log.items()}
        else:
            gate_note = {}

        return HubResponse(
            text=text,
            model=model,
            trace_id=trace_id,
            latency_ms=latency,
            controls_used=used_controls,
            safety={**safety, **({"akorn": "applied"} if gate_log else {}), **gate_note},
            confidence=confidence,
            uncertainty_reason=uncertainty_reason,
        )


@lru_cache(maxsize=4)
def _uncertainty_lexicon(locale: str) -> Dict[str, object]:
    normalized = (locale or "ja").lower()
    defaults: Dict[str, object] = {
        "line_template": "※推定信頼度: {confidence:.2f}（{confidence_label}） / 不確実要因: {reasons}",
        "reason_low": "低",
        "reason_join": ", ",
        "confidence_low_max": 0.54,
        "confidence_mid_max": 0.79,
        "confidence_label_low": "低",
        "confidence_label_mid": "中",
        "confidence_label_high": "高",
        "reason_labels": {
            "retrieval_sparse": "関連記憶が少ない",
            "retrieval_error": "記憶検索で一時的なエラー",
            "score_saturation_high": "スコア飽和が高い",
            "score_saturation_moderate": "スコア飽和がやや高い",
        },
    }
    if not normalized.startswith("ja"):
        return defaults
    path = Path(__file__).resolve().parents[2] / "locales" / "ja.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        section = payload.get("uncertainty_meta")
        if isinstance(section, dict):
            merged: Dict[str, object] = dict(defaults)
            for key in ("line_template", "reason_low", "reason_join"):
                val = section.get(key)
                if isinstance(val, str) and val.strip():
                    merged[key] = val
            labels = section.get("reason_labels")
            if isinstance(labels, dict):
                base_labels = dict(defaults["reason_labels"])  # type: ignore[arg-type]
                for code, label in labels.items():
                    if isinstance(code, str) and isinstance(label, str) and label.strip():
                        base_labels[code] = label
                merged["reason_labels"] = base_labels
            return merged
    except Exception:
        pass
    return defaults


def _confidence_label(confidence: float, lex: Dict[str, object]) -> str:
    try:
        low_max = float(lex.get("confidence_low_max", 0.54))
    except (TypeError, ValueError):
        low_max = 0.54
    try:
        mid_max = float(lex.get("confidence_mid_max", 0.79))
    except (TypeError, ValueError):
        mid_max = 0.79
    # Guard against invalid or reversed thresholds from runtime config.
    if not (0.0 <= low_max < mid_max <= 1.0):
        _LOGGER.warning(
            "llm_hub.confidence_threshold_fallback low=%s mid=%s -> default",
            low_max,
            mid_max,
        )
        low_max = 0.54
        mid_max = 0.79
    if confidence <= low_max:
        return str(lex.get("confidence_label_low", "low"))
    if confidence <= mid_max:
        return str(lex.get("confidence_label_mid", "mid"))
    return str(lex.get("confidence_label_high", "high"))
