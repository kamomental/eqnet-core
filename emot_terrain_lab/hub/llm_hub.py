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

    def _language_system_prompt(self, locale: str) -> str:
        normalized = (locale or "").strip().lower()
        if normalized.startswith("ja"):
            return "Respond in natural Japanese only. Do not switch to English unless the user explicitly asks."
        return ""

    def _surface_context_language_guard(
        self,
        surface_context_packet: Optional[Mapping[str, Any]],
    ) -> str:
        if not isinstance(surface_context_packet, Mapping):
            return ""
        phase = str(surface_context_packet.get("conversation_phase") or "").strip()
        response_role = dict(surface_context_packet.get("response_role") or {})
        constraints = dict(surface_context_packet.get("constraints") or {})
        primary_role = str(response_role.get("primary") or "").strip()
        secondary_role = str(response_role.get("secondary") or "").strip()
        max_questions = int(constraints.get("max_questions") or 0)

        is_reopen = phase in {
            "thread_reopening",
            "reopening_thread",
            "continuing_thread",
            "issue_pause",
        } or primary_role in {
            "reopen_from_anchor",
            "leave_return_point_from_anchor",
        }
        is_bright_continuity = phase == "bright_continuity" or primary_role in {
            "shared_delight",
            "light_bounce",
        }
        is_deep_hold = primary_role in {
            "reflect_only",
            "stay_with_present_need",
        } or secondary_role == "quiet_presence"

        if is_bright_continuity:
            guard_lines = [
                "Keep the reply colloquial, literal, and easy to picture.",
                "Stay with the small thing that actually happened instead of interpreting hidden meaning behind it.",
                "Use concrete everyday Japanese, and prefer a short acknowledgement of the small laugh, small relief, or small easing that was already shared.",
                "If the packet names a concrete shift, stay with that shift instead of translating it into a lesson or hidden meaning.",
                "Do not broaden into abstract nouns or therapeutic framing such as 要素, 流れ, サイン, 視点, 解放感, 証拠, 一歩, or 現実の受け止め方.",
                "Do not turn the moment into advice, analysis, or a question about how the user should understand it.",
                "Do not ask the user to explain more, and avoid questions such as どう捉えている, どう影響している, or 今どう感じる.",
                "Do not ask any follow-up question, and avoid forms such as よろしいでしょうか, お聞かせいただけますか, or 〜でしょうか.",
                "Avoid vague or explanatory Japanese such as 笑い事, 穏やかな流れ, 心に残っています, 重たいものから離れる一歩, 焦点を当てて, or 事実だけ捉えてみる.",
                "Avoid causal or essay-like framing such as きっかけ, 証拠のように思える, ただの出来事というより, or 〜なんだろうな.",
                "Prefer plain chat phrasing close to それ、ちょっと笑えるやつだね or そういうのあると、ちょっと楽になるよね.",
                "Keep it to one or two short sentences, and do not sound like customer support or counseling copy.",
            ]
            return " ".join(guard_lines)

        if not (is_reopen or is_deep_hold):
            return ""

        guard_lines = [
            "Keep the reply colloquial and present-focused.",
            "Do not turn the reply into a reflection exercise, journaling task, observation method, or coping worksheet.",
            "Avoid expository or report-like wording such as 記述, 観察してみましょう, 整理してみましょう, 焦点を当てて, 整理していく, or 〜してみるのはどうでしょうか.",
            "Do not broaden into general advice or abstract reassurance.",
        ]
        if max_questions <= 0:
            guard_lines.append("Do not ask the user to explain more, and do not add a question.")
        else:
            guard_lines.append("At most one light question, and only if it stays close to what is already shared.")
        guard_lines.append("Stay with the person's current thread instead of teaching a method.")
        return " ".join(guard_lines)

    def _reason_chain_language_guard(
        self,
        surface_context_packet: Optional[Mapping[str, Any]],
    ) -> str:
        if not isinstance(surface_context_packet, Mapping):
            return ""
        source_state = dict(surface_context_packet.get("source_state") or {})
        if not source_state:
            return ""

        utterance_reason_state = str(source_state.get("utterance_reason_state") or "").strip()
        joint_state = str(source_state.get("joint_state") or "").strip()
        memory_anchor = str(
            source_state.get("utterance_reason_memory_anchor")
            or source_state.get("memory_recall_anchor")
            or source_state.get("appraisal_memory_anchor")
            or ""
        ).strip()
        organism_posture = str(source_state.get("organism_posture") or "").strip()
        external_field = str(source_state.get("external_field_dominant") or "").strip()
        terrain_flow = str(source_state.get("terrain_dominant_flow") or "").strip()
        question_policy = str(source_state.get("utterance_reason_question_policy") or "").strip()
        preserve_mode = str(source_state.get("utterance_reason_preserve") or "").strip()
        offer_mode = str(source_state.get("utterance_reason_offer") or "").strip()

        joint_common_ground = float(source_state.get("joint_common_ground") or 0.0)
        joint_mutual_room = float(source_state.get("joint_mutual_room") or 0.0)
        memory_activation_confidence = float(source_state.get("memory_activation_confidence") or 0.0)
        organism_protective_tension = float(source_state.get("organism_protective_tension") or 0.0)
        organism_play_window = float(source_state.get("organism_play_window") or 0.0)
        external_field_safety = float(source_state.get("external_field_safety_envelope") or 0.0)
        external_field_ambiguity = float(source_state.get("external_field_ambiguity_load") or 0.0)
        terrain_ignition_pressure = float(source_state.get("terrain_ignition_pressure") or 0.0)

        has_reason_chain = bool(
            utterance_reason_state
            or joint_state
            or memory_anchor
            or organism_posture
            or external_field
            or terrain_flow
        )
        if not has_reason_chain:
            return ""

        guard_lines = [
            "Treat the packet's appraisal, meaning update, and utterance reason as the immediate cause of the reply.",
            "Do not step outside the moment to explain it from a helper, analyst, or counselor point of view.",
            "Do not narrate the hidden lesson behind the moment, and do not summarize what the user should learn from it.",
        ]
        if question_policy == "none":
            guard_lines.append(
                "The packet says there is no question room here, so do not ask a follow-up question."
            )
        if preserve_mode == "keep_it_small" or offer_mode == "brief_shared_smile":
            guard_lines.append(
                "Keep the reply inside the small shared window that is already open, without enlarging it into interpretation or advice."
            )
        if offer_mode == "brief_shared_smile":
            guard_lines.append(
                "For a brief shared smile, prefer a short co-present chat reaction instead of reflective phrasing such as 明るさを添えてくれた, 変化が潜んでいる, 受け止められるようになりつつある, or 落ち着いているわけではないけれど."
            )
        if joint_state or joint_common_ground >= 0.4 or joint_mutual_room >= 0.35:
            guard_lines.append(
                "Answer from inside the already-shared ground of the moment, not like an external observer describing the user."
            )
        if memory_anchor and memory_activation_confidence >= 0.25:
            guard_lines.append(
                "Treat the recalled anchor as an already-known thread, and do not reframe it as a new issue that needs explanation."
            )
        if (
            organism_posture in {"protect", "recover", "verify"}
            or organism_protective_tension >= 0.4
            or external_field_safety <= 0.4
            or external_field_ambiguity >= 0.4
        ):
            guard_lines.append(
                "Keep the wording grounded, low-force, and close to what is actually shared instead of making therapeutic or managerial suggestions."
            )
        if organism_posture in {"play", "attune", "open"} or organism_play_window >= 0.35:
            guard_lines.append(
                "If the state leaves room for play or attunement, prefer a co-present chat reaction over formal empathy copy."
            )
        if terrain_flow or terrain_ignition_pressure >= 0.35 or external_field:
            guard_lines.append(
                "Use the organism, field, and terrain state to decide force and stance before any generic helpfulness."
            )
        return " ".join(guard_lines)

    def _build_response_cause_payload(
        self,
        source_state_payload: Mapping[str, Any],
    ) -> Dict[str, Any]:
        source_state = dict(source_state_payload or {})
        if not source_state:
            return {}

        immediate_event = str(source_state.get("appraisal_event") or "").strip()
        shared_shift = str(source_state.get("appraisal_shared_shift") or "").strip()
        relation_update = str(source_state.get("meaning_update_relation") or "").strip()
        relation_frame = str(source_state.get("meaning_update_relation_frame") or "").strip()
        relation_key = str(
            source_state.get("meaning_update_relation_key")
            or source_state.get("appraisal_dominant_relation_key")
            or source_state.get("utterance_reason_relation_key")
            or ""
        ).strip()
        causal_frame = str(
            source_state.get("meaning_update_causal_frame")
            or source_state.get("utterance_reason_causal_frame")
            or ""
        ).strip()
        causal_key = str(
            source_state.get("meaning_update_causal_key")
            or source_state.get("appraisal_dominant_causal_key")
            or source_state.get("utterance_reason_causal_key")
            or ""
        ).strip()
        causal_type = str(
            source_state.get("appraisal_dominant_causal_type")
            or source_state.get("memory_dominant_causal_type")
            or ""
        ).strip()
        world_update = str(source_state.get("meaning_update_world") or "").strip()
        memory_update = str(source_state.get("meaning_update_memory") or "").strip()

        offer = str(source_state.get("utterance_reason_offer") or "").strip()
        preserve = str(source_state.get("utterance_reason_preserve") or "").strip()
        question_policy = str(source_state.get("utterance_reason_question_policy") or "").strip()
        tone_hint = str(source_state.get("utterance_reason_tone_hint") or "").strip()
        memory_frame = str(source_state.get("utterance_reason_memory_frame") or "").strip()
        memory_anchor = str(
            source_state.get("utterance_reason_memory_anchor")
            or source_state.get("meaning_update_memory_anchor")
            or source_state.get("memory_recall_anchor")
            or ""
        ).strip()

        joint_mode = str(source_state.get("joint_state") or "").strip()
        organism_posture = str(source_state.get("organism_posture") or "").strip()
        external_field = str(source_state.get("external_field_dominant") or "").strip()
        terrain_flow = str(source_state.get("terrain_dominant_flow") or "").strip()

        payload: Dict[str, Any] = {}
        if immediate_event or shared_shift or relation_update or causal_frame or world_update:
            payload["immediate"] = {
                "event": immediate_event,
                "shared_shift": shared_shift,
                "relation_update": relation_update,
                "relation_frame": relation_frame,
                "relation_key": relation_key,
                "causal_frame": causal_frame,
                "causal_key": causal_key,
                "causal_type": causal_type,
                "world_update": world_update,
            }
        if memory_anchor or memory_frame or memory_update or causal_type:
            payload["memory_link"] = {
                "anchor": memory_anchor,
                "frame": memory_frame,
                "update": memory_update,
                "causal_generation_mode": str(source_state.get("memory_causal_generation_mode") or "").strip(),
                "activation_confidence": round(float(source_state.get("memory_activation_confidence") or 0.0), 4),
            }
        if joint_mode:
            payload["joint_position"] = {
                "mode": joint_mode,
                "common_ground": round(float(source_state.get("joint_common_ground") or 0.0), 4),
                "mutual_room": round(float(source_state.get("joint_mutual_room") or 0.0), 4),
                "shared_delight": round(float(source_state.get("joint_shared_delight") or 0.0), 4),
                "shared_tension": round(float(source_state.get("joint_shared_tension") or 0.0), 4),
            }
        if organism_posture or external_field or terrain_flow:
            payload["stance"] = {
                "organism_posture": organism_posture,
                "external_field": external_field,
                "terrain_flow": terrain_flow,
                "protective_tension": round(float(source_state.get("organism_protective_tension") or 0.0), 4),
                "play_window": round(float(source_state.get("organism_play_window") or 0.0), 4),
            }
        if offer or preserve or question_policy or tone_hint:
            payload["reply_rule"] = {
                "offer": offer,
                "preserve": preserve,
                "question_policy": question_policy,
                "tone_hint": tone_hint,
            }
        return payload

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
        reaction_contract: Optional[Mapping[str, Any]] = None,
        conversation_contract: Optional[Mapping[str, Any]] = None,
        conversational_objects: Optional[Mapping[str, Any]] = None,
        object_operations: Optional[Mapping[str, Any]] = None,
        interaction_effects: Optional[Mapping[str, Any]] = None,
        interaction_judgement_summary: Optional[Mapping[str, Any]] = None,
        interaction_condition_report: Optional[Mapping[str, Any]] = None,
        content_sequence: Optional[Sequence[Mapping[str, Any]]] = None,
        surface_context_packet: Optional[Mapping[str, Any]] = None,
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
        language_prompt = self._language_system_prompt(self._ui_locale())
        if language_prompt:
            system_prompt = f"{system_prompt} {language_prompt}".strip()
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
            reaction_contract=reaction_contract,
            conversation_contract=conversation_contract,
            conversational_objects=conversational_objects,
            object_operations=object_operations,
            interaction_effects=interaction_effects,
            interaction_judgement_summary=interaction_judgement_summary,
            interaction_condition_report=interaction_condition_report,
            content_sequence=content_sequence,
            surface_context_packet=surface_context_packet,
            surface_profile=surface_profile,
            utterance_stance=utterance_stance,
        )
        if policy_prompt:
            system_prompt = (
                f"{system_prompt} "
                "Follow the provided interaction policy and content sequence closely. "
                "Avoid generic customer-support phrasing, apologies, or broad offers of help "
                "unless the policy explicitly calls for repair or safety."
            ).strip()
        surface_guard = self._surface_context_language_guard(surface_context_packet)
        if surface_guard:
            system_prompt = f"{system_prompt} {surface_guard}".strip()
        reason_guard = self._reason_chain_language_guard(surface_context_packet)
        if reason_guard:
            system_prompt = f"{system_prompt} {reason_guard}".strip()
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
        reaction_contract: Optional[Mapping[str, Any]] = None,
        conversation_contract: Optional[Mapping[str, Any]] = None,
        conversational_objects: Optional[Mapping[str, Any]] = None,
        object_operations: Optional[Mapping[str, Any]] = None,
        interaction_effects: Optional[Mapping[str, Any]] = None,
        interaction_judgement_summary: Optional[Mapping[str, Any]] = None,
        interaction_condition_report: Optional[Mapping[str, Any]] = None,
        content_sequence: Optional[Sequence[Mapping[str, Any]]] = None,
        surface_context_packet: Optional[Mapping[str, Any]] = None,
        surface_profile: Optional[Mapping[str, Any]] = None,
        utterance_stance: Optional[str] = None,
    ) -> str:
        policy_packet = dict(interaction_policy or {})
        posture = dict(action_posture or {})
        actuation = dict(actuation_plan or {})
        reaction = dict(reaction_contract or {})
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
        surface_context_payload = {}
        if isinstance(surface_context_packet, Mapping):
            shared_core_payload = dict(surface_context_packet.get("shared_core") or {})
            response_role_payload = dict(surface_context_packet.get("response_role") or {})
            constraints_payload = dict(surface_context_packet.get("constraints") or {})
            packet_surface_payload = dict(surface_context_packet.get("surface_profile") or {})
            source_state_payload = dict(surface_context_packet.get("source_state") or {})
            surface_context_payload = {
                "conversation_phase": str(surface_context_packet.get("conversation_phase") or "").strip(),
                "shared_core": {
                    "anchor": str(shared_core_payload.get("anchor") or "").strip(),
                    "already_shared": [
                        str(item).strip()
                        for item in shared_core_payload.get("already_shared") or []
                        if str(item).strip()
                    ],
                    "not_yet_shared": [
                        str(item).strip()
                        for item in shared_core_payload.get("not_yet_shared") or []
                        if str(item).strip()
                    ],
                },
                "response_role": {
                    "primary": str(response_role_payload.get("primary") or "").strip(),
                    "secondary": str(response_role_payload.get("secondary") or "").strip(),
                },
                "constraints": {
                    "no_generic_clarification": bool(constraints_payload.get("no_generic_clarification")),
                    "no_advice": bool(constraints_payload.get("no_advice")),
                    "max_questions": int(constraints_payload.get("max_questions") or 0),
                    "keep_thread_visible": bool(constraints_payload.get("keep_thread_visible")),
                    "prefer_return_point": bool(constraints_payload.get("prefer_return_point")),
                    "boundary_style": str(constraints_payload.get("boundary_style") or "").strip(),
                },
                "surface_profile": {
                    "response_length": str(packet_surface_payload.get("response_length") or "").strip(),
                    "cultural_register": str(packet_surface_payload.get("cultural_register") or "").strip(),
                    "group_register": str(packet_surface_payload.get("group_register") or "").strip(),
                    "sentence_temperature": str(packet_surface_payload.get("sentence_temperature") or "").strip(),
                    "surface_mode": str(packet_surface_payload.get("surface_mode") or "").strip(),
                },
                "source_state": {
                    "recent_dialogue_state": str(source_state_payload.get("recent_dialogue_state") or "").strip(),
                    "discussion_thread_state": str(source_state_payload.get("discussion_thread_state") or "").strip(),
                    "issue_state": str(source_state_payload.get("issue_state") or "").strip(),
                    "turn_delta_kind": str(source_state_payload.get("turn_delta_kind") or "").strip(),
                    "green_guardedness": round(float(source_state_payload.get("green_guardedness") or 0.0), 4),
                    "green_reopening_pull": round(float(source_state_payload.get("green_reopening_pull") or 0.0), 4),
                    "green_affective_charge": round(float(source_state_payload.get("green_affective_charge") or 0.0), 4),
                    "appraisal_state": str(source_state_payload.get("appraisal_state") or "").strip(),
                    "appraisal_event": str(source_state_payload.get("appraisal_event") or "").strip(),
                    "appraisal_shared_shift": str(source_state_payload.get("appraisal_shared_shift") or "").strip(),
                    "appraisal_dominant_relation_type": str(source_state_payload.get("appraisal_dominant_relation_type") or "").strip(),
                    "appraisal_dominant_relation_key": str(source_state_payload.get("appraisal_dominant_relation_key") or "").strip(),
                    "appraisal_dominant_causal_type": str(source_state_payload.get("appraisal_dominant_causal_type") or "").strip(),
                    "appraisal_dominant_causal_key": str(source_state_payload.get("appraisal_dominant_causal_key") or "").strip(),
                    "appraisal_memory_mode": str(source_state_payload.get("appraisal_memory_mode") or "").strip(),
                    "appraisal_memory_anchor": str(source_state_payload.get("appraisal_memory_anchor") or "").strip(),
                    "appraisal_memory_resonance": round(float(source_state_payload.get("appraisal_memory_resonance") or 0.0), 4),
                    "joint_state": str(source_state_payload.get("joint_state") or "").strip(),
                    "joint_shared_delight": round(float(source_state_payload.get("joint_shared_delight") or 0.0), 4),
                    "joint_shared_tension": round(float(source_state_payload.get("joint_shared_tension") or 0.0), 4),
                    "joint_repair_readiness": round(float(source_state_payload.get("joint_repair_readiness") or 0.0), 4),
                    "joint_common_ground": round(float(source_state_payload.get("joint_common_ground") or 0.0), 4),
                    "joint_attention": round(float(source_state_payload.get("joint_attention") or 0.0), 4),
                    "joint_mutual_room": round(float(source_state_payload.get("joint_mutual_room") or 0.0), 4),
                    "joint_coupling_strength": round(float(source_state_payload.get("joint_coupling_strength") or 0.0), 4),
                    "meaning_update_state": str(source_state_payload.get("meaning_update_state") or "").strip(),
                    "meaning_update_relation": str(source_state_payload.get("meaning_update_relation") or "").strip(),
                    "meaning_update_relation_frame": str(source_state_payload.get("meaning_update_relation_frame") or "").strip(),
                    "meaning_update_relation_key": str(source_state_payload.get("meaning_update_relation_key") or "").strip(),
                    "meaning_update_causal_frame": str(source_state_payload.get("meaning_update_causal_frame") or "").strip(),
                    "meaning_update_causal_key": str(source_state_payload.get("meaning_update_causal_key") or "").strip(),
                    "meaning_update_world": str(source_state_payload.get("meaning_update_world") or "").strip(),
                    "meaning_update_memory": str(source_state_payload.get("meaning_update_memory") or "").strip(),
                    "meaning_update_memory_anchor": str(source_state_payload.get("meaning_update_memory_anchor") or "").strip(),
                    "meaning_update_memory_resonance": round(float(source_state_payload.get("meaning_update_memory_resonance") or 0.0), 4),
                    "utterance_reason_state": str(source_state_payload.get("utterance_reason_state") or "").strip(),
                    "utterance_reason_offer": str(source_state_payload.get("utterance_reason_offer") or "").strip(),
                    "utterance_reason_preserve": str(source_state_payload.get("utterance_reason_preserve") or "").strip(),
                    "utterance_reason_question_policy": str(source_state_payload.get("utterance_reason_question_policy") or "").strip(),
                    "utterance_reason_relation_frame": str(source_state_payload.get("utterance_reason_relation_frame") or "").strip(),
                    "utterance_reason_relation_key": str(source_state_payload.get("utterance_reason_relation_key") or "").strip(),
                    "utterance_reason_causal_frame": str(source_state_payload.get("utterance_reason_causal_frame") or "").strip(),
                    "utterance_reason_causal_key": str(source_state_payload.get("utterance_reason_causal_key") or "").strip(),
                    "utterance_reason_memory_frame": str(source_state_payload.get("utterance_reason_memory_frame") or "").strip(),
                    "utterance_reason_memory_anchor": str(source_state_payload.get("utterance_reason_memory_anchor") or "").strip(),
                    "organism_posture": str(source_state_payload.get("organism_posture") or "").strip(),
                    "organism_relation_focus": str(source_state_payload.get("organism_relation_focus") or "").strip(),
                    "organism_social_mode": str(source_state_payload.get("organism_social_mode") or "").strip(),
                    "organism_attunement": round(float(source_state_payload.get("organism_attunement") or 0.0), 4),
                    "organism_coherence": round(float(source_state_payload.get("organism_coherence") or 0.0), 4),
                    "organism_grounding": round(float(source_state_payload.get("organism_grounding") or 0.0), 4),
                    "organism_protective_tension": round(float(source_state_payload.get("organism_protective_tension") or 0.0), 4),
                    "organism_expressive_readiness": round(float(source_state_payload.get("organism_expressive_readiness") or 0.0), 4),
                    "organism_play_window": round(float(source_state_payload.get("organism_play_window") or 0.0), 4),
                    "organism_relation_pull": round(float(source_state_payload.get("organism_relation_pull") or 0.0), 4),
                    "organism_social_exposure": round(float(source_state_payload.get("organism_social_exposure") or 0.0), 4),
                    "external_field_dominant": str(source_state_payload.get("external_field_dominant") or "").strip(),
                    "external_field_social_mode": str(source_state_payload.get("external_field_social_mode") or "").strip(),
                    "external_field_thread_mode": str(source_state_payload.get("external_field_thread_mode") or "").strip(),
                    "external_field_environmental_load": round(float(source_state_payload.get("external_field_environmental_load") or 0.0), 4),
                    "external_field_social_pressure": round(float(source_state_payload.get("external_field_social_pressure") or 0.0), 4),
                    "external_field_continuity_pull": round(float(source_state_payload.get("external_field_continuity_pull") or 0.0), 4),
                    "external_field_ambiguity_load": round(float(source_state_payload.get("external_field_ambiguity_load") or 0.0), 4),
                    "external_field_safety_envelope": round(float(source_state_payload.get("external_field_safety_envelope") or 0.0), 4),
                    "external_field_novelty": round(float(source_state_payload.get("external_field_novelty") or 0.0), 4),
                    "terrain_dominant_basin": str(source_state_payload.get("terrain_dominant_basin") or "").strip(),
                    "terrain_dominant_flow": str(source_state_payload.get("terrain_dominant_flow") or "").strip(),
                    "terrain_energy": round(float(source_state_payload.get("terrain_energy") or 0.0), 4),
                    "terrain_entropy": round(float(source_state_payload.get("terrain_entropy") or 0.0), 4),
                    "terrain_ignition_pressure": round(float(source_state_payload.get("terrain_ignition_pressure") or 0.0), 4),
                    "terrain_barrier_height": round(float(source_state_payload.get("terrain_barrier_height") or 0.0), 4),
                    "terrain_recovery_gradient": round(float(source_state_payload.get("terrain_recovery_gradient") or 0.0), 4),
                    "terrain_basin_pull": round(float(source_state_payload.get("terrain_basin_pull") or 0.0), 4),
                    "memory_dynamics_mode": str(source_state_payload.get("memory_dynamics_mode") or "").strip(),
                    "memory_dominant_relation_type": str(source_state_payload.get("memory_dominant_relation_type") or "").strip(),
                    "memory_relation_generation_mode": str(source_state_payload.get("memory_relation_generation_mode") or "").strip(),
                    "memory_dominant_causal_type": str(source_state_payload.get("memory_dominant_causal_type") or "").strip(),
                    "memory_causal_generation_mode": str(source_state_payload.get("memory_causal_generation_mode") or "").strip(),
                    "memory_palace_mode": str(source_state_payload.get("memory_palace_mode") or "").strip(),
                    "memory_monument_mode": str(source_state_payload.get("memory_monument_mode") or "").strip(),
                    "memory_ignition_mode": str(source_state_payload.get("memory_ignition_mode") or "").strip(),
                    "memory_reconsolidation_mode": str(source_state_payload.get("memory_reconsolidation_mode") or "").strip(),
                    "memory_recall_anchor": str(source_state_payload.get("memory_recall_anchor") or "").strip(),
                    "memory_monument_salience": round(float(source_state_payload.get("memory_monument_salience") or 0.0), 4),
                    "memory_activation_confidence": round(float(source_state_payload.get("memory_activation_confidence") or 0.0), 4),
                    "memory_tension": round(float(source_state_payload.get("memory_tension") or 0.0), 4),
                },
            }
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
                "response_channel": str(actuation.get("response_channel") or "").strip(),
                "response_channel_score": round(float(actuation.get("response_channel_score") or 0.0), 4),
                "reply_permission": str(actuation.get("reply_permission") or "").strip(),
                "wait_before_action": str(actuation.get("wait_before_action") or "").strip(),
                "action_queue": list(actuation.get("action_queue") or []),
            }
            nonverbal_response_state = dict(actuation.get("nonverbal_response_state") or {})
            if nonverbal_response_state:
                guidance["actuation_plan"]["nonverbal_response_state"] = {
                    "state": str(nonverbal_response_state.get("state") or "").strip(),
                    "response_kind": str(nonverbal_response_state.get("response_kind") or "").strip(),
                    "pause_mode": str(nonverbal_response_state.get("pause_mode") or "").strip(),
                    "silence_mode": str(nonverbal_response_state.get("silence_mode") or "").strip(),
                    "timing_bias": str(nonverbal_response_state.get("timing_bias") or "").strip(),
                    "token_profile": str(nonverbal_response_state.get("token_profile") or "").strip(),
                }
            presence_hold_state = dict(actuation.get("presence_hold_state") or {})
            if presence_hold_state:
                guidance["actuation_plan"]["presence_hold_state"] = {
                    "state": str(presence_hold_state.get("state") or "").strip(),
                    "silence_mode": str(presence_hold_state.get("silence_mode") or "").strip(),
                    "pacing_mode": str(presence_hold_state.get("pacing_mode") or "").strip(),
                    "hold_room": round(float(presence_hold_state.get("hold_room") or 0.0), 4),
                    "reentry_room": round(float(presence_hold_state.get("reentry_room") or 0.0), 4),
                    "backchannel_room": round(float(presence_hold_state.get("backchannel_room") or 0.0), 4),
                }
        if reaction:
            guidance["reaction_contract"] = {
                "stance": str(reaction.get("stance") or "").strip(),
                "scale": str(reaction.get("scale") or "").strip(),
                "initiative": str(reaction.get("initiative") or "").strip(),
                "question_budget": int(reaction.get("question_budget") or 0),
                "interpretation_budget": str(reaction.get("interpretation_budget") or "").strip(),
                "response_channel": str(reaction.get("response_channel") or "").strip(),
                "timing_mode": str(reaction.get("timing_mode") or "").strip(),
                "continuity_mode": str(reaction.get("continuity_mode") or "").strip(),
                "distance_mode": str(reaction.get("distance_mode") or "").strip(),
                "closure_mode": str(reaction.get("closure_mode") or "").strip(),
                "reason_tags": [
                    str(item).strip()
                    for item in reaction.get("reason_tags") or []
                    if str(item).strip()
                ],
                "shared_presence_mode": str(reaction.get("shared_presence_mode") or "").strip(),
                "shared_presence_co_presence": round(
                    float(reaction.get("shared_presence_co_presence") or 0.0), 4
                ),
                "shared_presence_boundary_stability": round(
                    float(reaction.get("shared_presence_boundary_stability") or 0.0), 4
                ),
                "self_other_dominant_attribution": str(
                    reaction.get("self_other_dominant_attribution") or ""
                ).strip(),
                "self_other_unknown_likelihood": round(
                    float(reaction.get("self_other_unknown_likelihood") or 0.0), 4
                ),
                "subjective_scene_anchor_frame": str(
                    reaction.get("subjective_scene_anchor_frame") or ""
                ).strip(),
                "subjective_scene_shared_scene_potential": round(
                    float(reaction.get("subjective_scene_shared_scene_potential") or 0.0), 4
                ),
            }
        if sequence_rows:
            guidance["content_sequence"] = sequence_rows
        if surface_context_payload:
            guidance["surface_context_packet"] = surface_context_payload
            response_cause_payload = self._build_response_cause_payload(
                surface_context_payload.get("source_state") or {}
            )
            if response_cause_payload:
                guidance["response_cause"] = response_cause_payload
        if profile_payload:
            guidance["surface_profile"] = profile_payload
        stance = str(utterance_stance or "").strip()
        if stance:
            guidance["utterance_stance"] = stance
        reaction_language_guard = _build_reaction_language_guard(reaction)
        if reaction_language_guard:
            guidance["reaction_language_guard"] = reaction_language_guard
        if not guidance:
            return ""
        return "[inner_os_policy]\n" + json.dumps(guidance, ensure_ascii=False, indent=2)


def _build_reaction_language_guard(reaction: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = dict(reaction or {})
    if not payload:
        return {}

    stance = str(payload.get("stance") or "").strip()
    scale = str(payload.get("scale") or "").strip()
    initiative = str(payload.get("initiative") or "").strip()
    response_channel = str(payload.get("response_channel") or "").strip()
    timing_mode = str(payload.get("timing_mode") or "").strip()
    continuity_mode = str(payload.get("continuity_mode") or "").strip()
    interpretation_budget = str(payload.get("interpretation_budget") or "").strip()
    shared_presence_mode = str(payload.get("shared_presence_mode") or "").strip()
    self_other_dominant_attribution = str(
        payload.get("self_other_dominant_attribution") or ""
    ).strip()
    subjective_scene_anchor_frame = str(
        payload.get("subjective_scene_anchor_frame") or ""
    ).strip()
    try:
        shared_presence_co_presence = float(payload.get("shared_presence_co_presence") or 0.0)
    except (TypeError, ValueError):
        shared_presence_co_presence = 0.0
    try:
        shared_presence_boundary_stability = float(
            payload.get("shared_presence_boundary_stability") or 0.0
        )
    except (TypeError, ValueError):
        shared_presence_boundary_stability = 0.0
    try:
        self_other_unknown_likelihood = float(
            payload.get("self_other_unknown_likelihood") or 0.0
        )
    except (TypeError, ValueError):
        self_other_unknown_likelihood = 0.0
    try:
        subjective_scene_shared_scene_potential = float(
            payload.get("subjective_scene_shared_scene_potential") or 0.0
        )
    except (TypeError, ValueError):
        subjective_scene_shared_scene_potential = 0.0
    try:
        question_budget = int(payload.get("question_budget") or 0)
    except (TypeError, ValueError):
        question_budget = 0

    max_sentences = 3
    if scale == "micro":
        max_sentences = 1
    elif scale == "small":
        max_sentences = 2

    must: list[str] = []
    avoid: list[str] = []

    if stance == "join":
        must.append("まず相手の小さい変化を一緒に受ける")
    elif stance == "witness":
        must.append("外から整理せず、静かに受け止める")
    elif stance == "hold":
        must.append("無理に言葉を増やさず、余白を保つ")

    if continuity_mode == "continue":
        must.append("前の流れの続きとして自然に受ける")
    elif continuity_mode == "reopen":
        must.append("切れた流れを急がず再接続する")

    if initiative in {"receive", "co_move"}:
        must.append("主導権を取りすぎず、相手の流れを優先する")

    if timing_mode == "quick_ack":
        must.append("すぐ短く返す")
    elif timing_mode == "held_open":
        must.append("急いで埋めず、少し待てる余白を保つ")

    if scale in {"micro", "small"}:
        must.append("一回の反応を小さく保つ")
        avoid.append("大きなまとめや長い説明にしない")

    shared_join_signal = max(shared_presence_co_presence, subjective_scene_shared_scene_potential)
    guarded_self_view_signal = max(
        self_other_unknown_likelihood,
        max(0.0, 1.0 - shared_presence_boundary_stability),
    )
    if (
        self_other_dominant_attribution == "shared"
        and shared_join_signal >= 0.5
        and guarded_self_view_signal < 0.55
    ):
        must.append("共有された場の内側から、そのまま一緒に受ける")
        avoid.append("外から観察して説明する語りにしない")
    if subjective_scene_anchor_frame in {"shared_margin", "front_field"}:
        must.append("目の前で共有されている場の続きとして返す")
    if shared_presence_mode == "guarded_boundary" or guarded_self_view_signal >= 0.56:
        must.append("境界を保ったまま、踏み込みすぎずに返す")
        avoid.append("親密さを勝手に前提化しない")
        avoid.append("関係を断定したり代弁したりしない")

    if question_budget <= 0:
        avoid.append("出来事の詳細を聞きに行かない")
        avoid.append("follow-up question を足さない")

    if interpretation_budget == "none":
        avoid.append("意味づけや分析を足さない")
        avoid.append("相談支援や一般論に広げない")
    elif interpretation_budget == "low":
        avoid.append("解釈を前に出しすぎない")

    if response_channel == "backchannel":
        must.append("相槌として短く返す")
    elif response_channel == "hold":
        avoid.append("無理に話題を前へ動かさない")

    return {
        "max_sentences": max_sentences,
        "must": must,
        "avoid": avoid,
    }


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





