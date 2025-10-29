# -*- coding: utf-8 -*-
"""
Reference helper:

Conversationテキストから「誰・いつ・どこ・何を」を推定し、MemoryPalaceの候補ノードを
軽量に検索して内的リプレイへ橋渡しする。

ライティング方針
-----------------
* すべての応答は「受容 → つながり確認 → Iメッセージ → 共同行動の芽」の順に整形する。
* 共同体感覚を損なう直接的なYouメッセージや「課題の分離」フレーズは phrasebook 側で除去する。
* LLM が不在でも Heuristic でフェイルセーフに動作する（テストはここを想定）。
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Iterable, List, Optional, Tuple

import numpy as np

from emot_terrain_lab.observer import generate_disclaimer
from emot_terrain_lab.persona.adler_phrasebook import clean_message, get_phrase


_LOCATION_HINTS = ("旅行", "旅", "出張", "散歩", "trip", "journey", "walk", "vacation")
_WHAT_HINTS = {
    "travel": ("旅行", "旅", "trip", "journey", "vacation"),
    "presentation": ("発表", "プレゼン", "presentation", "talk"),
    "conflict": ("喧嘩", "口論", "conflict", "argu", "けんか"),
}


@dataclass
class ReferenceResolution:
    who: Optional[str]
    when_year: Optional[int]
    where: Optional[str]
    what: Optional[str]
    summary: str


@dataclass
class ReplayCandidate:
    node_name: str
    label: str
    semantic: float
    affective: float
    anchor: float
    score: float


@dataclass
class ReplayOutcome:
    candidate: ReplayCandidate
    fidelity: float
    affect_strength: float


def resolve_reference(text: str) -> ReferenceResolution:
    summary = text.strip()
    who = None
    if re.search(r"(あなた|君|きみ|you)", text, re.IGNORECASE):
        who = "you"
    elif re.search(r"(私|わたし|ぼく|俺|I)\b", text, re.IGNORECASE):
        who = "self"

    year_match = re.search(r"(20\d{2}|19\d{2})年?", text)
    when_year = int(year_match.group(1)) if year_match else None

    tokens = re.findall(r"[A-Za-z\u3040-\u30ff\u4e00-\u9fff]+", text)
    where: Optional[str] = None
    for token in tokens:
        for hint in _LOCATION_HINTS:
            if hint in token:
                stripped = token.replace(hint, "")
                if stripped:
                    where = stripped
                break
        if where:
            break
        if token.endswith(("で", "へ")) and len(token) >= 2:
            where = token.rstrip("でへ")
            break

    what: Optional[str] = None
    lowered = text.lower()
    for label, hints in _WHAT_HINTS.items():
        if any(hint in text or hint in lowered for hint in hints):
            what = label
            break

    return ReferenceResolution(who=who, when_year=when_year, where=where, what=what, summary=summary)


def _semantic_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    ratio = SequenceMatcher(None, a.lower(), b.lower()).ratio()
    return float(np.clip(ratio, 0.0, 1.0))


def _anchor_score(label: str, ref: ReferenceResolution) -> float:
    score = 0.0
    if ref.when_year and str(ref.when_year) in label:
        score += 0.5
    if ref.where and ref.where in label:
        score += 0.3
    if ref.what and ref.what in label.lower():
        score += 0.2
    return float(np.clip(score, 0.0, 1.0))


def _affective_strength(trace: np.ndarray | None) -> float:
    if trace is None or trace.size == 0:
        return 0.0
    return float(np.clip(float(np.mean(trace)), 0.0, 1.0))


def search_memory(system, reference: ReferenceResolution, k: int = 3) -> List[ReplayCandidate]:
    palace = getattr(system, "memory_palace", None)
    if palace is None:
        return []

    candidates: List[ReplayCandidate] = []
    for node_name, labels in getattr(palace, "labels", {}).items():
        traces = getattr(palace, "traces", {}).get(node_name)
        affect = _affective_strength(traces)
        for label in labels:
            if not label:
                continue
            semantic = _semantic_similarity(label, reference.summary)
            anchor = _anchor_score(label, reference)
            score = 0.5 * semantic + 0.3 * affect + 0.2 * anchor
            if score <= 0.0:
                continue
            candidates.append(
                ReplayCandidate(
                    node_name=node_name,
                    label=label,
                    semantic=semantic,
                    affective=affect,
                    anchor=anchor,
                    score=score,
                )
            )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:k]


def run_replay(system, candidates: Iterable[ReplayCandidate]) -> Optional[ReplayOutcome]:
    executor = getattr(system, "replay_executor", None)
    best_outcome: Optional[ReplayOutcome] = None
    for candidate in candidates:
        fidelity: float
        affect_strength: float
        if executor and hasattr(executor, "run"):
            try:
                result = executor.run(candidate.node_name, candidate.label)
            except Exception:
                continue
            fidelity = float(result.get("fidelity", candidate.score))
            affect_strength = float(result.get("affect_strength", candidate.affective))
        else:
            # Heuristic fallback: use score and affect as pseudo fidelity.
            baseline = max(candidate.score, candidate.affective)
            fidelity = float(np.clip(0.4 + 0.6 * baseline, 0.0, 1.0))
            affect_strength = float(np.clip(candidate.affective, 0.0, 1.0))
        outcome = ReplayOutcome(candidate=candidate, fidelity=fidelity, affect_strength=affect_strength)
        if best_outcome is None or outcome.fidelity > best_outcome.fidelity:
            best_outcome = outcome
    return best_outcome


def _default_phrase(culture: str, tone: str, category: str, fallback: str, seed: str) -> str:
    phrase = get_phrase(culture, tone, category, seed=seed)
    if phrase:
        return phrase
    return fallback


def _co_agency_phrase(culture: str, tone: str, seed: str) -> Optional[str]:
    return get_phrase(culture, tone, "co_agency", seed=seed)


def _i_message_phrase(culture: str, tone: str, seed: str) -> Optional[str]:
    return get_phrase(culture, tone, "i_message", seed=seed)


def _truncate(reply: str, limit: Optional[int]) -> str:
    if limit is None or len(reply) <= limit:
        return reply
    if limit <= 1:
        return reply[:limit]
    return reply[: limit - 1].rstrip() + "…"


def compose_recall_response(
    outcome: ReplayOutcome,
    reference: ReferenceResolution,
    *,
    tone: str = "support",
    culture: str = "ja-JP",
    strategy: str = "recall",
    user_feedback: Optional[str] = None,
    max_reply_chars: Optional[int] = None,
) -> Tuple[str, float, str, str]:
    candidate = outcome.candidate
    fidelity = outcome.fidelity
    anchor = candidate.anchor

    seed = candidate.label or reference.summary or "memory"
    acceptance = _default_phrase(
        culture,
        tone,
        "acceptance",
        "そうだね、ここでいっしょに受け止めているよ。",
        seed,
    )
    belonging = get_phrase(culture, tone, "belonging", seed=seed)
    i_message = _i_message_phrase(culture, tone, seed) or ""
    co_phrase = _co_agency_phrase(culture, tone, seed)

    state = {
        "offer_gate": {
            "ses": float(np.clip(fidelity, 0.0, 1.0)),
            "threshold": 0.55,
            "suggestion_allowed": fidelity >= 0.55,
            "suppress_reason": "low_fidelity" if fidelity < 0.55 else "ok",
        },
        "evidence": {"eqnet_fields": {"fidelity": fidelity, "anchor": anchor}},
    }
    disclaimer = generate_disclaimer(state, tone=tone, culture=culture)
    source = state.get("disclaimer_source", "unknown")

    body_parts: List[str] = [acceptance]
    if belonging:
        body_parts.append(belonging)

    if strategy == "mend":
        if culture.lower().startswith("ja"):
            mend_text = f"{disclaimer} 私はその言葉を大事に受け止めるね。"
            if user_feedback:
                mend_text += f" いま教えてくれた『{user_feedback}』として記憶を整えるよ。"
            else:
                mend_text += " その気持ちをこれからも一緒に抱えていたいな。"
        else:
            mend_text = f"{disclaimer} I'm taking your correction to heart."
            if user_feedback:
                mend_text += f" I'll hold it as '{user_feedback}' so we stay aligned."
            else:
                mend_text += " I'll keep the feeling close to us."
        body_parts.append(mend_text)
    elif fidelity >= 0.65 and anchor >= 0.5:
        if culture.lower().startswith("ja"):
            body_parts.append(
                f"{disclaimer} 私は『{candidate.label}』の瞬間をはっきり覚えていて、"
                "その温度をいまも二人で抱いているよ。"
            )
        else:
            body_parts.append(
                f"{disclaimer} I remember '{candidate.label}' clearly and I'm holding that warmth with us."
            )
    elif fidelity >= 0.45:
        if culture.lower().startswith("ja"):
            body_parts.append(
                f"{disclaimer} 私は仮に『{candidate.label}』を思い出しているけれど、"
                "違っていたら教えてほしいな。"
            )
        else:
            body_parts.append(
                f"{disclaimer} I'm tentatively recalling '{candidate.label}'."
                " Please nudge me if I'm off."
            )
    else:
        if culture.lower().startswith("ja"):
            body_parts.append(
                f"{disclaimer} 私の記憶はまだぼんやりしているけれど、ここで一緒に少しずつ思い出したい。"
            )
        else:
            body_parts.append(
                f"{disclaimer} My memory is hazy, yet I'd love to rediscover it gently together."
            )

    if i_message:
        body_parts.append(i_message)
    if co_phrase:
        body_parts.append(co_phrase)

    reply = clean_message(culture, " ".join(part.strip() for part in body_parts if part))
    reply = _truncate(reply, max_reply_chars)
    return reply, fidelity, candidate.label, source


def handle_memory_reference(
    system,
    text: str,
    *,
    tone: str = "support",
    culture: str = "ja-JP",
    k: int = 3,
    strategy: str = "recall",
    user_feedback: Optional[str] = None,
    max_reply_chars: Optional[int] = None,
) -> dict:
    reference = resolve_reference(text)
    candidates = search_memory(system, reference, k=k)
    seed = reference.summary or text or "memory"

    if not candidates:
        state = {
            "offer_gate": {
                "ses": 0.0,
                "threshold": 0.6,
                "suggestion_allowed": False,
                "suppress_reason": "no_match",
            }
        }
        disclaimer = generate_disclaimer(state, tone=tone, culture=culture)
        parts = [
            get_phrase(culture, tone, "acceptance", seed=seed)
            or "そうだね、その気持ちをここで受け止めているよ。",
            get_phrase(culture, tone, "belonging", seed=seed),
            f"{disclaimer} その場面はまだはっきり記録に残っていないみたい。"
            " もしよければ少しずつ教えてもらえたらうれしいな。",
        ]
        reply = clean_message(culture, " ".join(part for part in parts if part))
        return {
            "reply": _truncate(reply, max_reply_chars),
            "fidelity": 0.0,
            "candidate": None,
            "meta": {"mode": strategy, "disclaimer_source": "template"},
        }

    outcome = run_replay(system, candidates)
    if outcome is None:
        reply = clean_message(
            culture,
            (
                get_phrase(culture, tone, "acceptance", seed=seed)
                or "そうだね、ここであなたと受け止めているよ。"
            )
            + " まだうまく思い出せなかった、ごめんね。",
        )
        return {
            "reply": _truncate(reply, max_reply_chars),
            "fidelity": 0.0,
            "candidate": None,
            "meta": {"mode": strategy, "disclaimer_source": "fallback"},
        }

    reply, fidelity, label, source = compose_recall_response(
        outcome,
        reference,
        tone=tone,
        culture=culture,
        strategy=strategy,
        user_feedback=user_feedback,
        max_reply_chars=max_reply_chars,
    )
    return {
        "reply": reply,
        "fidelity": fidelity,
        "candidate": {
            "node": outcome.candidate.node_name,
            "label": label,
            "score": outcome.candidate.score,
            "affective": outcome.candidate.affective,
        },
        "meta": {
            "mode": strategy,
            "anchor": outcome.candidate.anchor,
            "disclaimer_source": source,
        },
    }


__all__ = [
    "ReferenceResolution",
    "ReplayCandidate",
    "ReplayOutcome",
    "resolve_reference",
    "search_memory",
    "run_replay",
    "compose_recall_response",
    "handle_memory_reference",
]
