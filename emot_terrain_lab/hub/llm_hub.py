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
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
import time
import os

from terrain import llm as terrain_llm
from emot_terrain_lab.tools.registry import SkillRegistry
from emot_terrain_lab.hub.akorn import AkornGate, AkornConfig
from emot_terrain_lab.rag.lazy_rag import LazyRAG, LazyRAGConfig
from emot_terrain_lab.rag.sse_search import SSESearchAdapter
from emot_terrain_lab.vision.lmstudio_vlm import LMStudioVLMAdapter
from inner_os.conversation_contract import build_conversation_contract
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
    fallback_text: str = "I am sorry, but I could not prepare a response just now."


@dataclass
class HubResponse:
    text: str
    model: Optional[str]
    trace_id: str
    latency_ms: float
    controls_used: Dict[str, float]
    safety: Dict[str, str]
    model_source: str = ""
    confidence: float = 0.0
    uncertainty_reason: Tuple[str, ...] = ()
    retrieval_summary: Optional[Dict[str, object]] = None
    perception_summary: Optional[Dict[str, object]] = None


class LLMHub:
    """Very small routing facade over ``terrain.llm``."""

    def __init__(self, config: Optional[LLMHubConfig] = None) -> None:
        self.config = config or LLMHubConfig()
        self.registry = SkillRegistry()
        # Allow environment-based tuning of AKOrN small gains
        self.akorn = AkornGate(AkornConfig.from_env())
        self._lazy_rag: Optional[LazyRAG] = None
        self._sse_search: Optional[SSESearchAdapter] = None
        self._vlm_adapter: Optional[LMStudioVLMAdapter] = None
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

    def _get_sse_search(self) -> Optional[SSESearchAdapter]:
        if self._sse_search is None:
            adapter = SSESearchAdapter()
            if not adapter.enabled:
                return None
            self._sse_search = adapter
        return self._sse_search

    def _get_vlm_adapter(self) -> Optional[LMStudioVLMAdapter]:
        if self._vlm_adapter is None:
            adapter = LMStudioVLMAdapter()
            if not adapter.enabled:
                return None
            self._vlm_adapter = adapter
        return self._vlm_adapter

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
        image_path: Optional[str] = None,
        interaction_policy: Optional[Mapping[str, Any]] = None,
        action_posture: Optional[Mapping[str, Any]] = None,
        actuation_plan: Optional[Mapping[str, Any]] = None,
        conversation_contract: Optional[Mapping[str, Any]] = None,
        conversational_objects: Optional[Mapping[str, Any]] = None,
        object_operations: Optional[Mapping[str, Any]] = None,
        interaction_effects: Optional[Mapping[str, Any]] = None,
        interaction_judgement_summary: Optional[Mapping[str, Any]] = None,
        interaction_condition_report: Optional[Mapping[str, Any]] = None,
        content_sequence: Optional[Sequence[Mapping[str, Any]]] = None,
        surface_profile: Optional[Mapping[str, Any]] = None,
        utterance_stance: Optional[str] = None,
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
            Dict containing behaviour controls (pause_ms, temperature, warmth窶ｦ)
            produced by :class:`hub.policy.PolicyHead`.
        intent:
            Intent label (qa/chitchat/code窶ｦ). Currently only chitchat behaviour
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
        retrieval_summary: Optional[Dict[str, object]] = None
        perception_summary: Optional[Dict[str, object]] = None
        visual_context: Optional[str] = None
        if image_path:
            vlm = self._get_vlm_adapter()
            if vlm is not None:
                try:
                    perception_summary = vlm.describe_image(image_path, user_text=user_text)
                    visual_text = str(perception_summary.get("text") or "").strip() if perception_summary else ""
                    if visual_text:
                        visual_context = f"[vision]\n{visual_text}"
                except Exception as exc:
                    perception_summary = {"backend": "lmstudio_vlm", "image_path": image_path, "error": "vision_error", "detail": str(exc)}
        if not context:
            sse = self._get_sse_search()
            if sse is not None:
                try:
                    sse_hits = sse.search(user_text)
                    if sse_hits:
                        context = sse.build_context(user_text)
                        retrieval_summary = sse.summarize_hits(sse_hits)
                except Exception:
                    retrieval_error = True
                    context = None
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
                    if context:
                        retrieval_summary = {
                            "backend": "lazy_rag",
                            "context_chars": len(context),
                            "sat_ratio": sat_ratio,
                        }
        if visual_context:
            context = f"{visual_context}\n\n{context.strip()}" if context else visual_context
        prompt = user_text
        policy_prompt = self._build_inner_os_policy_prompt(
            interaction_policy=interaction_policy,
            action_posture=action_posture,
            actuation_plan=actuation_plan,
            conversation_contract=conversation_contract,
            conversational_objects=conversational_objects,
            object_operations=object_operations,
            interaction_effects=interaction_effects,
            interaction_judgement_summary=interaction_judgement_summary,
            interaction_condition_report=interaction_condition_report,
            content_sequence=content_sequence,
            surface_profile=surface_profile,
            utterance_stance=utterance_stance,
        )
        if context and policy_prompt:
            prompt = (
                context.strip()
                + "\n\n---\n\n"
                + policy_prompt
                + "\n\n---\n\n"
                + user_text.strip()
            )
        elif context:
            prompt = context.strip() + "\n\n---\n\n" + user_text.strip()
        elif policy_prompt:
            prompt = policy_prompt + "\n\n---\n\n" + user_text.strip()

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
        llm_info = terrain_llm.get_llm()
        model = llm_info.model or chosen_llm
        model_source = str(getattr(llm_info, "model_source", "") or "").strip()

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
            model_source=model_source,
            trace_id=trace_id,
            latency_ms=latency,
            controls_used=used_controls,
            safety={**safety, **({"akorn": "applied"} if gate_log else {}), **gate_note},
            confidence=confidence,
            uncertainty_reason=uncertainty_reason,
            retrieval_summary=retrieval_summary,
            perception_summary=perception_summary,
        )

    def _build_inner_os_policy_prompt(
        self,
        *,
        interaction_policy: Optional[Mapping[str, Any]] = None,
        action_posture: Optional[Mapping[str, Any]] = None,
        actuation_plan: Optional[Mapping[str, Any]] = None,
        conversation_contract: Optional[Mapping[str, Any]] = None,
        conversational_objects: Optional[Mapping[str, Any]] = None,
        object_operations: Optional[Mapping[str, Any]] = None,
        interaction_effects: Optional[Mapping[str, Any]] = None,
        interaction_judgement_summary: Optional[Mapping[str, Any]] = None,
        interaction_condition_report: Optional[Mapping[str, Any]] = None,
        content_sequence: Optional[Sequence[Mapping[str, Any]]] = None,
        surface_profile: Optional[Mapping[str, Any]] = None,
        utterance_stance: Optional[str] = None,
    ) -> str:
        policy_packet = dict(interaction_policy or {})
        posture = dict(action_posture or {})
        actuation = dict(actuation_plan or {})
        conversational_payload = dict(conversational_objects or {})
        operation_payload = dict(object_operations or {})
        effects_payload = dict(interaction_effects or {})
        judgement_summary_payload = dict(interaction_judgement_summary or {})
        condition_report_payload = dict(interaction_condition_report or {})
        contract_payload = dict(conversation_contract or {})
        sequence_rows = [
            {
                "act": str(item.get("act") or "").strip(),
                "text": str(item.get("text") or "").strip(),
            }
            for item in (content_sequence or [])
            if isinstance(item, Mapping) and str(item.get("text") or "").strip()
        ]
        profile_payload = {}
        if isinstance(surface_profile, Mapping):
            for key in (
                "opening_delay",
                "response_length",
                "sentence_temperature",
                "pause_insertion",
                "certainty_style",
                "opening_pace_windowed",
                "return_gaze_expectation",
                "voice_texture",
                "lightness_room",
                "continuity_weight",
            ):
                if key in {"lightness_room", "continuity_weight"}:
                    raw = surface_profile.get(key)
                    if raw is None:
                        continue
                    try:
                        profile_payload[key] = round(float(raw), 4)
                    except (TypeError, ValueError):
                        continue
                else:
                    value = str(surface_profile.get(key) or "").strip()
                    if value:
                        profile_payload[key] = value
        guidance: Dict[str, Any] = {}
        conversation_contract = contract_payload or _build_conversation_contract_payload(
            conversational_objects=conversational_payload,
            object_operations=operation_payload,
            interaction_effects=effects_payload,
            interaction_judgement_summary=judgement_summary_payload,
            interaction_condition_report=condition_report_payload,
            interaction_policy=policy_packet,
        )
        if conversation_contract:
            guidance["conversation_contract"] = conversation_contract
        if policy_packet:
            guidance["interaction_policy"] = {
                "dialogue_act": str(policy_packet.get("dialogue_act") or "").strip(),
                "opening_move": str(policy_packet.get("opening_move") or "").strip(),
                "followup_move": str(policy_packet.get("followup_move") or "").strip(),
                "closing_move": str(policy_packet.get("closing_move") or "").strip(),
                "dialogue_order": list(policy_packet.get("dialogue_order") or []),
                "do_not_cross": list(policy_packet.get("do_not_cross") or []),
            }
            ordered_operations = [
                str(item)
                for item in policy_packet.get("ordered_operation_kinds") or []
                if str(item).strip()
            ]
            if ordered_operations:
                guidance["interaction_policy"]["ordered_operations"] = ordered_operations
            ordered_effects = [
                str(item)
                for item in policy_packet.get("ordered_effect_kinds") or []
                if str(item).strip()
            ]
            if ordered_effects:
                guidance["interaction_policy"]["ordered_effects"] = ordered_effects
            shell_guidance = [str(item) for item in policy_packet.get("shell_guidance") or [] if str(item).strip()]
            if shell_guidance:
                guidance["interaction_policy"]["shell_guidance"] = shell_guidance
            expressive_style = dict(policy_packet.get("expressive_style_state") or {})
            if expressive_style:
                guidance["interaction_policy"]["expressive_style_state"] = {
                    "state": str(expressive_style.get("state") or "").strip(),
                    "lightness_room": round(float(expressive_style.get("lightness_room") or 0.0), 4),
                    "continuity_weight": round(float(expressive_style.get("continuity_weight") or 0.0), 4),
                    "winner_margin": round(float(expressive_style.get("winner_margin") or 0.0), 4),
                }
        if posture:
            guidance["action_posture"] = {
                "engagement_mode": str(posture.get("engagement_mode") or "").strip(),
                "outcome_goal": str(posture.get("outcome_goal") or "").strip(),
                "boundary_mode": str(posture.get("boundary_mode") or "").strip(),
                "attention_target": str(posture.get("attention_target") or "").strip(),
                "memory_write_priority": str(posture.get("memory_write_priority") or "").strip(),
            }
        if actuation:
            guidance["actuation_plan"] = {
                "execution_mode": str(actuation.get("execution_mode") or "").strip(),
                "primary_action": str(actuation.get("primary_action") or "").strip(),
                "reply_permission": str(actuation.get("reply_permission") or "").strip(),
                "wait_before_action": str(actuation.get("wait_before_action") or "").strip(),
                "action_queue": list(actuation.get("action_queue") or []),
            }
        if sequence_rows:
            guidance["content_sequence"] = sequence_rows
        if profile_payload:
            guidance["surface_profile"] = profile_payload
        stance = str(utterance_stance or "").strip()
        if stance:
            guidance["utterance_stance"] = stance
        if not guidance:
            return ""
        return "[inner_os_policy]\n" + json.dumps(guidance, ensure_ascii=False, indent=2)


def _build_conversation_contract_payload(
    *,
    conversational_objects: Mapping[str, Any],
    object_operations: Mapping[str, Any],
    interaction_effects: Mapping[str, Any],
    interaction_judgement_summary: Mapping[str, Any],
    interaction_condition_report: Mapping[str, Any],
    interaction_policy: Mapping[str, Any],
) -> Dict[str, Any]:
    return build_conversation_contract(
        conversational_objects=conversational_objects,
        object_operations=object_operations,
        interaction_effects=interaction_effects,
        interaction_judgement_summary=interaction_judgement_summary,
        interaction_condition_report=interaction_condition_report,
        interaction_policy=interaction_policy,
    )


@lru_cache(maxsize=4)
def _uncertainty_lexicon(locale: str) -> Dict[str, object]:
    normalized = (locale or "ja").lower()
    defaults: Dict[str, object] = {
        "line_template": "confidence: {confidence:.2f} ({confidence_label}) / reasons: {reasons}",
        "reason_low": "low",
        "reason_join": ", ",
        "confidence_low_max": 0.54,
        "confidence_mid_max": 0.79,
        "confidence_label_low": "low",
        "confidence_label_mid": "mid",
        "confidence_label_high": "high",
        "reason_labels": {
            "retrieval_sparse": "retrieval_sparse",
            "retrieval_error": "retrieval_error",
            "score_saturation_high": "score_saturation_high",
            "score_saturation_moderate": "score_saturation_moderate",
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





