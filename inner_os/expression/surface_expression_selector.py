from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class SurfaceExpressionCandidate:
    text: str
    formality: float = 0.5
    warmth: float = 0.5
    youthfulness: float = 0.5
    intimacy: float = 0.5
    genericity: float = 0.5
    consultation_tone: float = 0.5
    brevity: float = 0.5
    preferred_registers: tuple[str, ...] = field(default_factory=tuple)
    blocked_registers: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SurfaceExpressionProfile:
    target_formality: float
    target_warmth: float
    target_youthfulness: float
    target_intimacy: float
    genericity_penalty: float
    consultation_penalty: float
    brevity_preference: float
    register_name: str = ""


def choose_surface_expression(
    candidates: Sequence[SurfaceExpressionCandidate],
    *,
    cultural_register: str = "",
    group_register: str = "",
    sentence_temperature: str = "",
    mode: str = "",
    recent_history: Iterable[str] = (),
) -> str:
    cleaned = [candidate for candidate in candidates if candidate.text.strip()]
    if not cleaned:
        return ""

    profile = _derive_surface_expression_profile(
        cultural_register=cultural_register,
        group_register=group_register,
        sentence_temperature=sentence_temperature,
        mode=mode,
    )
    history = tuple(str(item or "").strip() for item in recent_history if str(item or "").strip())

    best_text = cleaned[0].text
    best_score = float("-inf")
    for index, candidate in enumerate(cleaned):
        score = _score_candidate(candidate, profile, history=history)
        # Keep earlier ordering as a stable tiebreaker.
        score -= index * 0.001
        if score > best_score:
            best_score = score
            best_text = candidate.text
    return best_text


def build_surface_expression_candidates(
    texts: Sequence[str],
    *,
    candidate_profile: str = "",
) -> list[SurfaceExpressionCandidate]:
    profile_name = str(candidate_profile or "").strip()
    if profile_name == "thread_reopen_return":
        return _build_thread_reopen_return_candidates(texts)
    if profile_name == "deep_reflection_stay":
        return _build_deep_reflection_stay_candidates(texts)
    if profile_name == "deep_reflection_presence":
        return _build_deep_reflection_presence_candidates(texts)
    if profile_name == "continuity_opening":
        return _build_continuity_opening_candidates(texts)
    if profile_name == "thread_presence":
        return _build_thread_presence_candidates(texts)
    if profile_name == "quiet_presence":
        return _build_quiet_presence_candidates(texts)
    if profile_name == "stay_with_present_need":
        return _build_stay_with_present_need_candidates(texts)
    if profile_name == "light_question":
        return _build_light_question_candidates(texts)
    return [SurfaceExpressionCandidate(text=str(text or "").strip()) for text in texts if str(text or "").strip()]


def _derive_surface_expression_profile(
    *,
    cultural_register: str,
    group_register: str,
    sentence_temperature: str,
    mode: str,
) -> SurfaceExpressionProfile:
    register = str(cultural_register or "").strip()
    group = str(group_register or "").strip()
    temperature = str(sentence_temperature or "").strip()
    current_mode = str(mode or "").strip()

    if register in {"careful_polite", "public_courteous", "hierarchy_respectful"}:
        return SurfaceExpressionProfile(
            target_formality=0.85,
            target_warmth=0.45,
            target_youthfulness=0.18,
            target_intimacy=0.28,
            genericity_penalty=0.28,
            consultation_penalty=0.22,
            brevity_preference=0.52,
            register_name=register or "formal",
        )

    warmth = 0.72 if temperature in {"warm", "gentle"} else 0.58
    intimacy = 0.68 if group in {"one_to_one", "threaded_group"} else 0.52
    youthfulness = 0.64 if register in {"casual_shared", "soft_companion", ""} else 0.48
    if current_mode == "live_mode":
        warmth = max(warmth, 0.68)
        intimacy = min(intimacy + 0.04, 0.82)
    elif current_mode == "love_mode":
        warmth = min(warmth + 0.08, 0.9)
        intimacy = min(intimacy + 0.12, 0.92)

    return SurfaceExpressionProfile(
        target_formality=0.24,
        target_warmth=warmth,
        target_youthfulness=youthfulness,
        target_intimacy=intimacy,
        genericity_penalty=0.42,
        consultation_penalty=0.5,
        brevity_preference=0.72,
        register_name=register or "casual_shared",
    )


def _score_candidate(
    candidate: SurfaceExpressionCandidate,
    profile: SurfaceExpressionProfile,
    *,
    history: Sequence[str],
) -> float:
    if profile.register_name and profile.register_name in candidate.blocked_registers:
        return float("-inf")

    score = 0.0
    if candidate.preferred_registers:
        if profile.register_name in candidate.preferred_registers:
            score += 0.18
        else:
            score -= 0.08

    score -= abs(candidate.formality - profile.target_formality) * 0.45
    score -= abs(candidate.warmth - profile.target_warmth) * 0.32
    score -= abs(candidate.youthfulness - profile.target_youthfulness) * 0.22
    score -= abs(candidate.intimacy - profile.target_intimacy) * 0.26
    score -= candidate.genericity * profile.genericity_penalty
    score -= candidate.consultation_tone * profile.consultation_penalty
    score += candidate.brevity * profile.brevity_preference * 0.18

    if any(candidate.text in item for item in history):
        score -= 0.24
    return score


def _build_thread_reopen_return_candidates(texts: Sequence[str]) -> list[SurfaceExpressionCandidate]:
    candidates: list[SurfaceExpressionCandidate] = []
    for index, text in enumerate(texts):
        body = str(text or "").strip()
        if not body:
            continue
        if index == 0:
            candidates.append(
                SurfaceExpressionCandidate(
                    text=body,
                    formality=0.18,
                    warmth=0.76,
                    youthfulness=0.72,
                    intimacy=0.7,
                    genericity=0.18,
                    consultation_tone=0.06,
                    brevity=0.92,
                    preferred_registers=("casual_shared", "soft_companion"),
                )
            )
        elif index == 1:
            candidates.append(
                SurfaceExpressionCandidate(
                    text=body,
                    formality=0.24,
                    warmth=0.68,
                    youthfulness=0.58,
                    intimacy=0.62,
                    genericity=0.22,
                    consultation_tone=0.08,
                    brevity=0.86,
                )
            )
        else:
            candidates.append(
                SurfaceExpressionCandidate(
                    text=body,
                    formality=0.56,
                    warmth=0.58,
                    youthfulness=0.26,
                    intimacy=0.46,
                    genericity=0.32,
                    consultation_tone=0.18,
                    brevity=0.64,
                    preferred_registers=("careful_polite", "public_courteous"),
                )
            )
    return candidates


def _build_deep_reflection_stay_candidates(texts: Sequence[str]) -> list[SurfaceExpressionCandidate]:
    candidates: list[SurfaceExpressionCandidate] = []
    for index, text in enumerate(texts):
        body = str(text or "").strip()
        if not body:
            continue
        if index == 0:
            candidates.append(
                SurfaceExpressionCandidate(
                    text=body,
                    formality=0.16,
                    warmth=0.84,
                    youthfulness=0.74,
                    intimacy=0.72,
                    genericity=0.14,
                    consultation_tone=0.04,
                    brevity=0.88,
                    preferred_registers=("casual_shared", "soft_companion"),
                )
            )
        else:
            candidates.append(
                SurfaceExpressionCandidate(
                    text=body,
                    formality=0.28 if index == 1 else 0.34,
                    warmth=0.76 if index == 1 else 0.7,
                    youthfulness=0.54 if index == 1 else 0.46,
                    intimacy=0.66 if index == 1 else 0.58,
                    genericity=0.18 if index == 1 else 0.22,
                    consultation_tone=0.06 if index == 1 else 0.08,
                    brevity=0.84,
                )
            )
    return candidates


def _build_deep_reflection_presence_candidates(texts: Sequence[str]) -> list[SurfaceExpressionCandidate]:
    candidates: list[SurfaceExpressionCandidate] = []
    for index, text in enumerate(texts):
        body = str(text or "").strip()
        if not body:
            continue
        candidates.append(
            SurfaceExpressionCandidate(
                text=body,
                formality=0.2 + min(index, 2) * 0.06,
                warmth=0.8 - min(index, 2) * 0.05,
                youthfulness=0.68 - min(index, 2) * 0.08,
                intimacy=0.68 - min(index, 2) * 0.06,
                genericity=0.16 + min(index, 2) * 0.04,
                consultation_tone=0.04 + min(index, 2) * 0.03,
                brevity=0.86,
                preferred_registers=("casual_shared", "soft_companion") if index == 0 else (),
            )
        )
    return candidates


def _build_continuity_opening_candidates(texts: Sequence[str]) -> list[SurfaceExpressionCandidate]:
    candidates: list[SurfaceExpressionCandidate] = []
    for index, text in enumerate(texts):
        body = str(text or "").strip()
        if not body:
            continue
        if index == 0:
            candidates.append(
                SurfaceExpressionCandidate(
                    text=body,
                    formality=0.18,
                    warmth=0.8,
                    youthfulness=0.76,
                    intimacy=0.68,
                    genericity=0.18,
                    consultation_tone=0.04,
                    brevity=0.72,
                    preferred_registers=("casual_shared", "soft_companion"),
                )
            )
        else:
            candidates.append(
                SurfaceExpressionCandidate(
                    text=body,
                    formality=0.34 + min(index, 2) * 0.08,
                    warmth=0.72 - min(index, 2) * 0.04,
                    youthfulness=0.56 - min(index, 2) * 0.08,
                    intimacy=0.58 - min(index, 2) * 0.04,
                    genericity=0.2 + min(index, 2) * 0.04,
                    consultation_tone=0.06 + min(index, 2) * 0.03,
                    brevity=0.78,
                )
            )
    return candidates


def _build_thread_presence_candidates(texts: Sequence[str]) -> list[SurfaceExpressionCandidate]:
    candidates: list[SurfaceExpressionCandidate] = []
    for index, text in enumerate(texts):
        body = str(text or "").strip()
        if not body:
            continue
        candidates.append(
            SurfaceExpressionCandidate(
                text=body,
                formality=0.22 + min(index, 2) * 0.06,
                warmth=0.78 - min(index, 2) * 0.04,
                youthfulness=0.66 - min(index, 2) * 0.07,
                intimacy=0.7 - min(index, 2) * 0.05,
                genericity=0.16 + min(index, 2) * 0.05,
                consultation_tone=0.04 + min(index, 2) * 0.03,
                brevity=0.82,
                preferred_registers=("casual_shared", "soft_companion") if index == 0 else (),
            )
        )
    return candidates


def _build_quiet_presence_candidates(texts: Sequence[str]) -> list[SurfaceExpressionCandidate]:
    candidates: list[SurfaceExpressionCandidate] = []
    for index, text in enumerate(texts):
        body = str(text or "").strip()
        if not body:
            continue
        candidates.append(
            SurfaceExpressionCandidate(
                text=body,
                formality=0.24 + min(index, 2) * 0.08,
                warmth=0.82 - min(index, 2) * 0.05,
                youthfulness=0.7 - min(index, 2) * 0.1,
                intimacy=0.72 - min(index, 2) * 0.05,
                genericity=0.16 + min(index, 2) * 0.05,
                consultation_tone=0.05 + min(index, 2) * 0.04,
                brevity=0.74 if index == 0 else 0.82,
                preferred_registers=("casual_shared", "soft_companion") if index > 0 else (),
            )
        )
    return candidates


def _build_stay_with_present_need_candidates(texts: Sequence[str]) -> list[SurfaceExpressionCandidate]:
    candidates: list[SurfaceExpressionCandidate] = []
    for index, text in enumerate(texts):
        body = str(text or "").strip()
        if not body:
            continue
        candidates.append(
            SurfaceExpressionCandidate(
                text=body,
                formality=0.3 if index == 0 else 0.16 + min(index, 2) * 0.06,
                warmth=0.72 if index == 0 else 0.84 - min(index, 2) * 0.06,
                youthfulness=0.42 if index == 0 else 0.74 - min(index, 2) * 0.12,
                intimacy=0.56 if index == 0 else 0.74 - min(index, 2) * 0.06,
                genericity=0.18 if index == 0 else 0.14 + min(index, 2) * 0.04,
                consultation_tone=0.08 if index == 0 else 0.04 + min(index, 2) * 0.03,
                brevity=0.64 if index == 0 else 0.88,
                preferred_registers=("casual_shared", "soft_companion") if index > 0 else (),
            )
        )
    return candidates


def _build_light_question_candidates(texts: Sequence[str]) -> list[SurfaceExpressionCandidate]:
    candidates: list[SurfaceExpressionCandidate] = []
    for index, text in enumerate(texts):
        body = str(text or "").strip()
        if not body:
            continue
        candidates.append(
            SurfaceExpressionCandidate(
                text=body,
                formality=0.38 if index == 0 else 0.22 + min(index, 2) * 0.05,
                warmth=0.68 if index == 0 else 0.8 - min(index, 2) * 0.05,
                youthfulness=0.34 if index == 0 else 0.66 - min(index, 2) * 0.08,
                intimacy=0.5 if index == 0 else 0.66 - min(index, 2) * 0.04,
                genericity=0.2 if index == 0 else 0.16 + min(index, 2) * 0.04,
                consultation_tone=0.1 if index == 0 else 0.06 + min(index, 2) * 0.03,
                brevity=0.66 if index == 0 else 0.8,
                preferred_registers=("casual_shared", "soft_companion") if index > 0 else (),
            )
        )
    return candidates
