from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional


QUESTION_MARKERS = ("?", "what", "why", "how", "which", "where", "when")
UNCERTAINTY_MARKERS = ("maybe", "unclear", "not sure", "still", "yet", "perhaps")


@dataclass
class WorkingMemorySnapshot:
    focus_text: str = ""
    focus_anchor: str = ""
    current_focus: str = "ambient"
    unresolved_count: int = 0
    open_loops: List[str] = field(default_factory=list)
    carryover_load: float = 0.0
    pending_meaning: float = 0.0
    social_focus: float = 0.0
    bodily_salience: float = 0.0
    memory_pressure: float = 0.0
    semantic_seed_focus: str = ""
    semantic_seed_anchor: str = ""
    semantic_seed_strength: float = 0.0
    semantic_seed_recurrence: float = 0.0
    long_term_theme_focus: str = ""
    long_term_theme_anchor: str = ""
    long_term_theme_kind: str = ""
    long_term_theme_summary: str = ""
    long_term_theme_strength: float = 0.0
    conscious_residue_focus: str = ""
    conscious_residue_anchor: str = ""
    conscious_residue_summary: str = ""
    conscious_residue_strength: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WorkingMemoryCore:
    """A thin short-horizon focus layer between live sensing and long memory."""

    def snapshot(
        self,
        *,
        user_input: Mapping[str, Any] | None,
        sensor_input: Mapping[str, Any] | None,
        current_state: Mapping[str, Any] | None,
        relational_world: Mapping[str, Any] | None,
        previous_trace: Mapping[str, Any] | None = None,
        recall_payload: Mapping[str, Any] | None = None,
    ) -> WorkingMemorySnapshot:
        user_text = str((user_input or {}).get("text") or "").strip()
        current_state = dict(current_state or {})
        previous_trace = dict(previous_trace or {})
        relational_world = dict(relational_world or {})
        sensor_input = dict(sensor_input or {})
        recall_payload = dict(recall_payload or {})
        semantic_seed_focus = str(
            current_state.get("semantic_seed_focus")
            or previous_trace.get("semantic_seed_focus")
            or ""
        ).strip()
        semantic_seed_anchor = str(
            current_state.get("semantic_seed_anchor")
            or previous_trace.get("semantic_seed_anchor")
            or ""
        ).strip()
        semantic_seed_strength = _float_from(
            current_state,
            "semantic_seed_strength",
            _float_from(previous_trace, "semantic_seed_strength", 0.0),
        )
        semantic_seed_recurrence = _float_from(
            current_state,
            "semantic_seed_recurrence",
            _float_from(previous_trace, "semantic_seed_recurrence", 0.0),
        )
        long_term_theme_focus = str(
            current_state.get("long_term_theme_focus")
            or previous_trace.get("long_term_theme_focus")
            or ""
        ).strip()
        long_term_theme_anchor = str(
            current_state.get("long_term_theme_anchor")
            or previous_trace.get("long_term_theme_anchor")
            or ""
        ).strip()
        long_term_theme_kind = str(
            current_state.get("long_term_theme_kind")
            or previous_trace.get("long_term_theme_kind")
            or ""
        ).strip()
        long_term_theme_summary = str(
            current_state.get("long_term_theme_summary")
            or previous_trace.get("long_term_theme_summary")
            or ""
        ).strip()
        long_term_theme_strength = _float_from(
            current_state,
            "long_term_theme_strength",
            _float_from(previous_trace, "long_term_theme_strength", 0.0),
        )
        conscious_residue_focus = str(
            current_state.get("conscious_residue_focus")
            or previous_trace.get("conscious_residue_focus")
            or ""
        ).strip()
        conscious_residue_anchor = str(
            current_state.get("conscious_residue_anchor")
            or previous_trace.get("conscious_residue_anchor")
            or ""
        ).strip()
        conscious_residue_summary = str(
            current_state.get("conscious_residue_summary")
            or previous_trace.get("conscious_residue_summary")
            or ""
        ).strip()
        conscious_residue_strength = _float_from(
            current_state,
            "conscious_residue_strength",
            _float_from(previous_trace, "conscious_residue_strength", 0.0),
        )

        focus_text = user_text or str(previous_trace.get("focus_text") or recall_payload.get("summary") or semantic_seed_focus or long_term_theme_focus or conscious_residue_focus or "").strip()
        focus_anchor = (
            str(relational_world.get("place_memory_anchor") or "").strip()
            or str(current_state.get("memory_anchor") or "").strip()
            or str(previous_trace.get("focus_anchor") or "").strip()
            or semantic_seed_anchor
            or long_term_theme_anchor
            or conscious_residue_anchor
            or _anchor_from_text(focus_text)
        )
        open_loops = _extract_open_loops(user_text, recall_payload=recall_payload)
        previous_unresolved = int(_float_from(previous_trace, "unresolved_count", 0.0))
        unresolved_count = max(len(open_loops), previous_unresolved if not open_loops and previous_unresolved > 0 else 0)

        person_count = int((sensor_input or {}).get("person_count", current_state.get("person_count", 0)) or 0)
        body_stress = _float_from(sensor_input, "body_stress_index", _float_from(current_state, "stress", 0.0))
        near_body_risk = _float_from(current_state, "near_body_risk", 0.0)
        affiliation_bias = _float_from(current_state, "affiliation_bias", 0.45)
        temporal_pressure = _float_from(current_state, "temporal_pressure", 0.0)
        recalled_tentative_bias = _float_from(current_state, "recalled_tentative_bias", _float_from(recall_payload, "tentative_bias", 0.0))

        private_flag = str((sensor_input or {}).get("body_state_flag") or current_state.get("body_state_flag") or "")
        pending_meaning = _clamp01(
            recalled_tentative_bias
            + (0.18 if str(recall_payload.get("reinterpretation_mode") or "") == "grounding_deferral" else 0.0)
            + min(unresolved_count, 3) * 0.12
            + semantic_seed_strength * 0.08
            + long_term_theme_strength * 0.06
            + conscious_residue_strength * 0.04
        )
        social_focus = _clamp01(person_count * 0.16 + affiliation_bias * 0.24 + (0.16 if "social_role" in relational_world else 0.0))
        bodily_salience = _clamp01(body_stress * 0.46 + near_body_risk * 0.24 + (0.22 if private_flag == "private_high_arousal" else 0.0))
        previous_load = _float_from(previous_trace, "carryover_load", _float_from(current_state, "working_memory_pressure", 0.0))
        memory_pressure = _clamp01(
            previous_load * 0.42
            + pending_meaning * 0.34
            + min(unresolved_count, 3) * 0.08
            + temporal_pressure * 0.14
            + social_focus * 0.08
            + bodily_salience * 0.12
            + semantic_seed_strength * 0.08
            + long_term_theme_strength * 0.06
            + conscious_residue_strength * 0.05
        )
        carryover_load = _clamp01(previous_load * 0.7 + memory_pressure * 0.38 + semantic_seed_strength * 0.1 + long_term_theme_strength * 0.08 + conscious_residue_strength * 0.06)
        current_focus = _select_focus(
            focus_anchor=focus_anchor,
            pending_meaning=pending_meaning,
            social_focus=social_focus,
            bodily_salience=bodily_salience,
            unresolved_count=unresolved_count,
        )

        return WorkingMemorySnapshot(
            focus_text=focus_text[:240],
            focus_anchor=focus_anchor[:160],
            current_focus=current_focus,
            unresolved_count=unresolved_count,
            open_loops=open_loops[:4],
            carryover_load=round(carryover_load, 4),
            pending_meaning=round(pending_meaning, 4),
            social_focus=round(social_focus, 4),
            bodily_salience=round(bodily_salience, 4),
            memory_pressure=round(memory_pressure, 4),
            semantic_seed_focus=semantic_seed_focus[:120],
            semantic_seed_anchor=semantic_seed_anchor[:160],
            semantic_seed_strength=round(semantic_seed_strength, 4),
            semantic_seed_recurrence=round(semantic_seed_recurrence, 4),
            long_term_theme_focus=long_term_theme_focus[:120],
            long_term_theme_anchor=long_term_theme_anchor[:160],
            long_term_theme_kind=long_term_theme_kind[:80],
            long_term_theme_summary=long_term_theme_summary[:160],
            long_term_theme_strength=round(long_term_theme_strength, 4),
            conscious_residue_focus=conscious_residue_focus[:120],
            conscious_residue_anchor=conscious_residue_anchor[:160],
            conscious_residue_summary=conscious_residue_summary[:160],
            conscious_residue_strength=round(conscious_residue_strength, 4),
        )

    def settle_after_turn(
        self,
        *,
        snapshot: WorkingMemorySnapshot,
        reply_text: str,
        current_state: Mapping[str, Any] | None = None,
        recall_payload: Mapping[str, Any] | None = None,
    ) -> WorkingMemorySnapshot:
        current_state = dict(current_state or {})
        recall_payload = dict(recall_payload or {})
        answered = bool(str(reply_text or "").strip())
        pending_meaning = float(snapshot.pending_meaning)
        unresolved_count = int(snapshot.unresolved_count)
        if answered:
            pending_meaning = max(0.0, pending_meaning - 0.12)
            if unresolved_count > 0 and "?" not in reply_text and pending_meaning < 0.34:
                unresolved_count -= 1
        if str(recall_payload.get("reinterpretation_mode") or "") == "grounding_deferral":
            pending_meaning = max(pending_meaning, 0.28)
        carryover_load = _clamp01(snapshot.carryover_load * 0.82 + pending_meaning * 0.22 + unresolved_count * 0.06)
        memory_pressure = _clamp01(snapshot.memory_pressure * 0.72 + carryover_load * 0.22 + _float_from(current_state, "temporal_pressure", 0.0) * 0.08)
        current_focus = _select_focus(
            focus_anchor=snapshot.focus_anchor,
            pending_meaning=pending_meaning,
            social_focus=snapshot.social_focus,
            bodily_salience=snapshot.bodily_salience,
            unresolved_count=unresolved_count,
        )
        return WorkingMemorySnapshot(
            focus_text=snapshot.focus_text,
            focus_anchor=snapshot.focus_anchor,
            current_focus=current_focus,
            unresolved_count=unresolved_count,
            open_loops=list(snapshot.open_loops),
            carryover_load=round(carryover_load, 4),
            pending_meaning=round(pending_meaning, 4),
            social_focus=round(snapshot.social_focus, 4),
            bodily_salience=round(snapshot.bodily_salience, 4),
            memory_pressure=round(memory_pressure, 4),
            semantic_seed_focus=snapshot.semantic_seed_focus,
            semantic_seed_anchor=snapshot.semantic_seed_anchor,
            semantic_seed_strength=round(snapshot.semantic_seed_strength, 4),
            semantic_seed_recurrence=round(snapshot.semantic_seed_recurrence, 4),
            long_term_theme_focus=snapshot.long_term_theme_focus,
            long_term_theme_anchor=snapshot.long_term_theme_anchor,
            long_term_theme_kind=snapshot.long_term_theme_kind,
            long_term_theme_summary=snapshot.long_term_theme_summary,
            long_term_theme_strength=round(snapshot.long_term_theme_strength, 4),
            conscious_residue_focus=snapshot.conscious_residue_focus,
            conscious_residue_anchor=snapshot.conscious_residue_anchor,
            conscious_residue_summary=snapshot.conscious_residue_summary,
            conscious_residue_strength=round(snapshot.conscious_residue_strength, 4),
        )

    def build_trace_record(
        self,
        *,
        snapshot: WorkingMemorySnapshot,
        current_state: Mapping[str, Any] | None,
        relational_world: Mapping[str, Any] | None,
    ) -> Dict[str, Any]:
        current_state = dict(current_state or {})
        relational_world = dict(relational_world or {})
        return {
            "kind": "working_memory_trace",
            "summary": f"{snapshot.current_focus}:{snapshot.focus_anchor or snapshot.focus_text[:32]}",
            "text": snapshot.focus_text or snapshot.focus_anchor,
            "memory_anchor": snapshot.focus_anchor or str(current_state.get("memory_anchor") or "working-memory"),
            "culture_id": relational_world.get("culture_id"),
            "community_id": relational_world.get("community_id"),
            "social_role": relational_world.get("social_role"),
            "current_focus": snapshot.current_focus,
            "focus_text": snapshot.focus_text,
            "focus_anchor": snapshot.focus_anchor,
            "unresolved_count": snapshot.unresolved_count,
            "open_loops": list(snapshot.open_loops),
            "carryover_load": snapshot.carryover_load,
            "pending_meaning": snapshot.pending_meaning,
            "social_focus": snapshot.social_focus,
            "bodily_salience": snapshot.bodily_salience,
            "memory_pressure": snapshot.memory_pressure,
            "semantic_seed_focus": snapshot.semantic_seed_focus,
            "semantic_seed_anchor": snapshot.semantic_seed_anchor,
            "semantic_seed_strength": snapshot.semantic_seed_strength,
            "semantic_seed_recurrence": snapshot.semantic_seed_recurrence,
            "long_term_theme_focus": snapshot.long_term_theme_focus,
            "long_term_theme_anchor": snapshot.long_term_theme_anchor,
            "long_term_theme_kind": snapshot.long_term_theme_kind,
            "long_term_theme_summary": snapshot.long_term_theme_summary,
            "long_term_theme_strength": snapshot.long_term_theme_strength,
            "conscious_residue_focus": snapshot.conscious_residue_focus,
            "conscious_residue_anchor": snapshot.conscious_residue_anchor,
            "conscious_residue_summary": snapshot.conscious_residue_summary,
            "conscious_residue_strength": snapshot.conscious_residue_strength,
        }


def _extract_open_loops(text: str, *, recall_payload: Mapping[str, Any]) -> List[str]:
    lowered = str(text or "").strip().lower()
    loops: List[str] = []
    if any(marker in lowered for marker in QUESTION_MARKERS):
        loops.append("question")
    if any(marker in lowered for marker in UNCERTAINTY_MARKERS):
        loops.append("uncertainty")
    if str(recall_payload.get("reinterpretation_mode") or "") == "grounding_deferral":
        loops.append("deferred_meaning")
    if _float_from(recall_payload, "tentative_bias", 0.0) >= 0.28:
        loops.append("tentative_reading")
    return loops


def _select_focus(
    *,
    focus_anchor: str,
    pending_meaning: float,
    social_focus: float,
    bodily_salience: float,
    unresolved_count: int,
) -> str:
    if pending_meaning >= 0.4 or unresolved_count > 0:
        return "meaning"
    if bodily_salience >= max(0.4, social_focus + 0.08):
        return "body"
    if social_focus >= 0.38:
        return "social"
    if focus_anchor:
        return "place"
    return "ambient"


def _anchor_from_text(text: str) -> str:
    parts = [part.strip() for part in str(text or "").replace("\n", " ").split(" ") if part.strip()]
    return " ".join(parts[:6])[:160]


def _float_from(mapping: Mapping[str, Any], key: str, default: float) -> float:
    try:
        return float(mapping.get(key, default))
    except (TypeError, ValueError, AttributeError):
        return float(default)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
