from __future__ import annotations


def resolve_partner_utterance_stance(
    *,
    relation_bias_strength: float,
    related_person_ids: list[str],
    partner_address_hint: str = "",
    partner_timing_hint: str = "",
    partner_stance_hint: str = "",
) -> str:
    """相手軸ヒントから発話スタンスを共通決定する。"""
    if not related_person_ids or relation_bias_strength < 0.28:
        return "neutral_observation"
    if partner_timing_hint == "delayed" or partner_stance_hint == "respectful":
        return "measured_check_in"
    if partner_address_hint == "companion" and partner_stance_hint == "familiar":
        return "warm_check_in"
    return "gentle_check_in"


def relation_episode_naming_from_stance(utterance_stance: str, social_interpretation: str = "") -> str:
    """発話と同じ軸で relation episode の呼び名をそろえる。"""
    if utterance_stance == "warm_check_in":
        return "warm_reconnection"
    if utterance_stance == "measured_check_in":
        return "measured_reapproach"
    if utterance_stance == "gentle_check_in":
        return "gentle_recontact"
    if social_interpretation:
        return "social_contact_trace"
    return "neutral_observation_trace"
