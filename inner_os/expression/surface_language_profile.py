from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from emot_terrain_lab.i18n.locale import normalize_locale


@dataclass(frozen=True)
class SurfaceLanguageProfile:
    banter_move: str
    lexical_variation_mode: str
    group_register: str
    dominant_inputs: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "banter_move": self.banter_move,
            "lexical_variation_mode": self.lexical_variation_mode,
            "group_register": self.group_register,
            "dominant_inputs": list(self.dominant_inputs),
        }


def derive_surface_language_profile(
    *,
    recovery_state: str,
    protection_mode_name: str,
    grice_state: str,
    expressive_style_name: str,
    expressive_style_history_focus: str,
    relational_continuity_name: str,
    relational_banter_style: str,
    relational_lexical_variation_bias: float,
    relational_banter_room: float,
    lightness_budget_name: str,
    lightness_banter_room: float,
    lightness_playful_ceiling: float,
    lightness_suppression: float,
    social_topology_name: str,
    cultural_state_name: str,
    cultural_joke_ratio_ceiling: float,
    lexical_variation_carry_bias: float,
    identity_arc_kind: str = "",
    identity_arc_phase: str = "",
    identity_arc_open_tension: str = "",
) -> SurfaceLanguageProfile:
    identity_arc_kind = str(identity_arc_kind or "").strip()
    identity_arc_phase = str(identity_arc_phase or "").strip()
    identity_arc_open_tension = str(identity_arc_open_tension or "").strip()
    identity_repair_thread = identity_arc_kind in {"repairing_bond", "holding_thread", "shared_place_thread"}
    identity_stabilizing = identity_arc_kind in {"stabilizing_self", "repairing_self"}
    identity_forward_edge = identity_arc_kind == "growing_edge" and identity_arc_phase in {"holding", "integrating"}
    identity_timing_sensitive = identity_arc_open_tension in {
        "timing_sensitive_reentry",
        "careful_repair",
        "guarded_closeness",
    }
    public_or_hierarchy = social_topology_name in {"public_visible", "hierarchical"}
    threaded_group = social_topology_name == "threaded_group"
    playful_open = (
        recovery_state == "open"
        and protection_mode_name not in {"contain", "stabilize", "shield"}
        and grice_state not in {"hold_obvious_advice", "attune_without_repeating"}
        and lightness_budget_name == "open_play"
        and lightness_banter_room >= 0.34
        and lightness_playful_ceiling >= 0.3
        and lightness_suppression < 0.32
        and not (identity_stabilizing and identity_timing_sensitive)
    )
    lexical_open = (
        relational_lexical_variation_bias >= 0.34
        or lexical_variation_carry_bias >= 0.1
    )

    banter_move = "none"
    lexical_variation_mode = "plain"
    group_register = social_topology_name or "ambient"
    dominant_inputs: list[str] = []

    if public_or_hierarchy:
        lexical_variation_mode = "formal_measured"
        if cultural_joke_ratio_ceiling >= 0.12 and relational_banter_style in {
            "respectful_light",
            "soft_formal",
        }:
            banter_move = "respectful_light"
            dominant_inputs.append("public_banter_restraint")
        dominant_inputs.append("public_or_hierarchy")
    elif threaded_group:
        group_register = "threaded_group"
        lexical_variation_mode = "group_attuned" if (
            lexical_open or cultural_state_name == "group_attuned"
        ) else "plain"
        if (
            lightness_budget_name in {"warm_only", "open_play"}
            and (
                expressive_style_name in {"warm_companion", "grounded_gentle", "light_playful"}
                or identity_repair_thread
            )
            and relational_banter_room >= 0.22
        ):
            banter_move = "thread_soften"
            dominant_inputs.append("threaded_group_attunement")
        if identity_repair_thread:
            dominant_inputs.append("identity_repair_thread")
        dominant_inputs.append("threaded_group")
    elif playful_open:
        group_register = "one_to_one"
        if identity_repair_thread:
            banter_move = "warm_refrain"
            dominant_inputs.append("identity_repair_warm_refrain")
        elif relational_banter_style in {"gentle_tease", "compact_wit"}:
            banter_move = relational_banter_style
            dominant_inputs.append(f"relational_banter_{relational_banter_style}")
        elif relational_banter_style == "warm_refrain":
            banter_move = "warm_refrain"
            dominant_inputs.append("relational_banter_warm_refrain")
        elif (
            expressive_style_name == "warm_companion"
            or expressive_style_history_focus == "warm_companion"
            or relational_continuity_name in {"reopening", "co_regulating"}
        ):
            banter_move = "warm_refrain"
            dominant_inputs.append("warm_continuity_refrain")

        if identity_repair_thread or identity_timing_sensitive:
            lexical_variation_mode = "warm_varied" if lexical_open else "plain"
            if lexical_variation_mode == "warm_varied":
                dominant_inputs.append("identity_sensitive_lexical")
        elif relational_banter_style == "compact_wit":
            lexical_variation_mode = "compact_varied"
            dominant_inputs.append("compact_wit_lexical")
        elif lexical_open or banter_move in {"gentle_tease", "warm_refrain"}:
            lexical_variation_mode = "warm_varied"
            dominant_inputs.append("warm_lexical_open")
    else:
        if identity_repair_thread and not public_or_hierarchy and lightness_budget_name in {"warm_only", "open_play"}:
            banter_move = "warm_refrain"
            dominant_inputs.append("identity_repair_warm_refrain")
            if lexical_open:
                lexical_variation_mode = "warm_varied"
                dominant_inputs.append("identity_repair_lexical")
        elif (
            expressive_style_name == "warm_companion"
            and lightness_budget_name in {"warm_only", "open_play"}
            and recovery_state == "open"
            and cultural_state_name not in {"public_courteous", "hierarchy_respectful"}
        ):
            banter_move = "warm_refrain"
            dominant_inputs.append("warm_companion_refrain")
        if lexical_open and lightness_budget_name in {"warm_only", "open_play"}:
            lexical_variation_mode = "warm_varied"
            dominant_inputs.append("ambient_lexical_open")
        if identity_stabilizing:
            banter_move = "none"
            lexical_variation_mode = "plain"
            dominant_inputs.append("identity_stabilizing_plain")
        elif identity_forward_edge and not public_or_hierarchy and lightness_budget_name in {"warm_only", "open_play"}:
            if lexical_open:
                lexical_variation_mode = "warm_varied"
                dominant_inputs.append("identity_forward_edge_lexical")

    return SurfaceLanguageProfile(
        banter_move=banter_move,
        lexical_variation_mode=lexical_variation_mode,
        group_register=group_register,
        dominant_inputs=dominant_inputs,
    )


def shape_surface_language_text(
    text: str,
    *,
    surface_profile: dict[str, Any],
    locale: str | None = None,
) -> str:
    body = str(text or "").strip()
    if not body:
        return ""
    normalized_locale = normalize_locale(locale or "en")
    is_ja = normalized_locale.startswith("ja")

    banter_move = str(surface_profile.get("banter_move") or "").strip()
    lexical_variation_mode = str(surface_profile.get("lexical_variation_mode") or "").strip()
    group_register = str(surface_profile.get("group_register") or "").strip()
    cultural_register = str(surface_profile.get("cultural_register") or "").strip()
    sentence_temperature = str(surface_profile.get("sentence_temperature") or "").strip()
    if not banter_move and lexical_variation_mode in {"", "plain"}:
        if is_ja:
            softened = _soften_japanese_surface_register(
                body,
                cultural_register=cultural_register,
                group_register=group_register,
                sentence_temperature=sentence_temperature,
            )
            return softened
        return body

    prefix, core = _split_leading_prefix(body, locale=normalized_locale)
    if not core:
        return body

    if not is_ja and lexical_variation_mode in {"warm_varied", "compact_varied"}:
        core = _apply_contractions(core)
    elif not is_ja and lexical_variation_mode == "formal_measured":
        core = _expand_respect_marker(core)

    if is_ja:
        if banter_move == "warm_refrain" and not _starts_with_any(core, ("ここは一緒に見ていけます。", "ここにいます。")):
            core = f"ここは一緒に見ていけます。{core}"
        elif banter_move == "thread_soften" and not _starts_with_any(core, ("この流れのままで、", "いまの流れのままで、")):
            core = f"この流れのままで、{core}"
        elif (
            banter_move == "respectful_light"
            and group_register in {"public_visible", "hierarchical"}
            and not _starts_with_any(core, ("丁寧に言うと、", "いまは、"))
        ):
            core = f"丁寧に言うと、{core}"
    else:
        if banter_move == "warm_refrain" and not _starts_with_any(
            core,
            ("I'm with you here.", "I am with you here.", "I'm here with you.", "I am here with you."),
        ):
            core = f"I'm with you here. {core}"
        elif banter_move == "gentle_tease" and not _starts_with_any(core, ("Just enough,", "Just lightly,")):
            core = f"Just enough, {core}"
        elif banter_move == "compact_wit" and not _starts_with_any(core, ("Short version:", "Briefly,")):
            core = f"Short version: {core}"
        elif banter_move == "thread_soften" and not _starts_with_any(core, ("For this thread,", "For now,")):
            core = f"For this thread, {core}"
        elif (
            banter_move == "respectful_light"
            and group_register in {"public_visible", "hierarchical"}
            and not _starts_with_any(core, ("Respectfully,", "Carefully,"))
        ):
            core = f"Respectfully, {core}"

    shaped = f"{prefix}{core}".strip()
    if is_ja:
        shaped = _soften_japanese_surface_register(
            shaped,
            cultural_register=cultural_register,
            group_register=group_register,
            sentence_temperature=sentence_temperature,
        )
    return shaped


def _split_leading_prefix(text: str, *, locale: str = "en") -> tuple[str, str]:
    body = str(text or "").strip()
    if not body:
        return "", ""
    if str(locale or "").startswith("ja"):
        prefixes = ("… ", "... ", ".. ", "いまは、", "丁寧に言うと、", "ここにいます。")
    else:
        prefixes = ("... ", ".. ", "Carefully, ", "Gently, ", "Respectfully, ")
    for marker in prefixes:
        if body.startswith(marker):
            return marker, body[len(marker) :].strip()
    return "", body


def _starts_with_any(text: str, prefixes: tuple[str, ...]) -> bool:
    lowered = str(text or "")
    return any(lowered.startswith(prefix) for prefix in prefixes)


def _apply_contractions(text: str) -> str:
    updated = str(text or "")
    replacements = (
        ("I am ", "I'm "),
        ("I will ", "I'll "),
        ("we will ", "we'll "),
        ("We will ", "We'll "),
        ("do not ", "don't "),
        ("Do not ", "Don't "),
        ("it is ", "it's "),
        ("It is ", "It's "),
        ("that is ", "that's "),
        ("That is ", "That's "),
    )
    for before, after in replacements:
        updated = updated.replace(before, after)
    return updated


def _soften_japanese_surface_register(
    text: str,
    *,
    cultural_register: str,
    group_register: str,
    sentence_temperature: str,
) -> str:
    body = str(text or "").strip()
    if not body:
        return ""
    if cultural_register in {"careful_polite", "public_courteous", "hierarchy_respectful"}:
        return body
    if group_register not in {"ambient", "one_to_one", "threaded_group", ""}:
        return body
    if sentence_temperature not in {"warm", "gentle", ""}:
        return body

    softened = body
    replacements = (
        ("大丈夫です。", "大丈夫だよ。"),
        ("大丈夫です", "大丈夫だよ"),
        ("いいです。", "いいよ。"),
        ("いいです", "いいよ"),
        ("かまいません。", "かまわないよ。"),
        ("かまいません", "かまわないよ"),
        ("話せます。", "話せるよ。"),
        ("話せます", "話せるよ"),
        ("付き添えます。", "付き添えるよ。"),
        ("付き添えます", "付き添えるよ"),
        ("進めます。", "進めるよ。"),
        ("進めます", "進めるよ"),
    )
    for source, target in replacements:
        softened = softened.replace(source, target)
    return softened


def _expand_respect_marker(text: str) -> str:
    body = str(text or "")
    if body.startswith("Carefully,") or body.startswith("Respectfully,"):
        return body
    if body.startswith("Short version:"):
        return body.replace("Short version:", "Briefly,", 1)
    return body
