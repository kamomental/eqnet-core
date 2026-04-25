from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

from .speech_act_contract import SpeechActAnalysis, speech_act_analysis_from_dict


_QUESTION_PHRASES: tuple[str, ...] = (
    "ですか",
    "ますか",
    "でしょうか",
    "いかがですか",
    "どうでしたか",
    "どうですか",
    "ありますか",
    "教えて",
    "聞かせて",
    "話して",
)

_UNCERTAINTY_META_PHRASES: tuple[str, ...] = (
    "推定信頼度",
    "不確実要因",
)

_ASSISTANT_ATTRACTOR_PHRASES: tuple[str, ...] = (
    "さて、さっきの話の続きですね",
    "お疲れ様です",
    "お疲れ様でした",
    "お久しぶりですね",
    "お気持ち",
    "どんな出来事があったのか",
    "どんなことですか",
    "ぜひ教えて",
    "聞かせてください",
    "お聞かせいただけますか",
    "気兼ねなく",
    "今の状態をそのまま受け入れて",
    "あなたのペースで",
    "ゆっくり休んで",
    "過ごしてみてください",
    "体調管理も忘れずに",
    "その後の様子はどうでしょうか",
    "よろしければ",
)

_INTERPRETIVE_BRIGHT_PHRASES: tuple[str, ...] = (
    "きっかけ",
    "証拠",
    "ように思える",
    "のでしょうね",
    "かもしれませんね",
    "ただの出来事というより",
    "少し和らげ",
    "心にも光が差し込",
    "重たいものから離れる一歩",
    "現実にも向き合",
    "受け止め方",
    "どう影響しているのか",
    "今の感覚は",
    "気持ちをどう捉えて",
    "少し軽くなったのかもしれません",
    "心が少し緩んだ",
    "心が少し軽くなった証拠",
    "解放感",
    "今の不安やしんどさ",
    "少し楽になったサイン",
)

_ELICITATION_PHRASES: tuple[str, ...] = (
    "見守ってください",
    "観察して",
    "振り返って",
    "整理して",
    "受け止めてください",
    "深く考えずに",
    "共有してください",
    "描写して",
    "焦点を当てて",
    "触れても良い",
    "ゆっくりと見守って",
    "今の感覚を",
)

_SPLIT_MARKERS: tuple[str, ...] = ("。", "！", "!", "？", "?", "\n")

_ASSISTANT_ATTRACTOR_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(?:おっしゃる通り|さて|昨日の続き|先ほどの(?:会話|話)|さっきの話の続き)"),
    re.compile(r"^(?:お疲れ様(?:です|でした)|お久しぶりですね)"),
    re.compile(r"(?:よろしければ|聞かせて(?:ください|いただけますか)|教えていただけますか)"),
    re.compile(r"(?:ぜひ|気兼ねなく|あなたのペース|体調管理も忘れずに)"),
    re.compile(r"(?:今はどう(?:お過ごし|されて)|その後はどう(?:でした|でしょう|されて))"),
)

_INTERPRETIVE_BRIGHT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:きっかけ|証拠|サイン|解放感)"),
    re.compile(r"(?:ように思(?:える|えます)|かもしれません|のでしょう|んだろう)"),
    re.compile(r"(?:心が少し(?:緩んだ|軽くなった)|少し(?:軽やかな気分|楽になった))"),
    re.compile(r"(?:和らげ|重たいものから離れる|心の中に溜まっていく)"),
    re.compile(r"(?:どう捉えて|どう影響して|受け止め方|気持ちをどう)"),
)

_ELICITATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:見守って|観察して|振り返って|整理して|受け止めてください)"),
    re.compile(r"(?:深く考えずに|共有してください|描写して|焦点を当てて|触れても良い)"),
)

_SMALL_SHARED_REACTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:ふふ|笑える|笑えた|笑っちゃう)"),
    re.compile(r"(?:ちょっと楽になる|少し気が楽になる|和む|ほっとする)"),
)

_INFORMATION_REQUEST_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:教えて|聞かせて|話して|共有して|描写して)"),
    re.compile(r"(?:何|なに|なぜ|どうして|どんな|どのよう).{0,16}(?:ですか|ますか|でしょうか|でしたか|かな)"),
    re.compile(r"(?:どう|どんな).{0,16}(?:感じ|気持ち|出来事|こと|流れ)"),
)

_INTERPRETATION_ACT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:つまり|要するに|ということ|という意味)"),
    re.compile(r"(?:かもしれません|かもしれない|のでしょう|んだろう|ように思(?:える|えます)|ように見える)"),
    re.compile(r"(?:意味|影響|理由|原因|証拠|サイン|きっかけ|受け止め方)"),
    re.compile(r"(?:気持ち|心|感覚).{0,12}(?:変化|軽く|緩ん|彩って|表れて)"),
)

_HEAVY_SURFACE_ACT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:整理しましょう|受け入れて|休んで|体調管理|あなたのペース)"),
    re.compile(r"(?:見守って|観察して|振り返って|深く考え|焦点を当てて)"),
    re.compile(r"(?:一つの入り口|大切です|必要があります|してみてください)"),
)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _source_state(
    surface_context_packet: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(surface_context_packet, Mapping):
        return {}
    source_state = surface_context_packet.get("source_state")
    return dict(source_state) if isinstance(source_state, Mapping) else {}


def _constraints(
    surface_context_packet: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(surface_context_packet, Mapping):
        return {}
    constraints = surface_context_packet.get("constraints")
    return dict(constraints) if isinstance(constraints, Mapping) else {}


def _reaction_contract(
    reaction_contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(reaction_contract, Mapping):
        return {}
    return dict(reaction_contract)


def _split_sentences(text: str) -> list[str]:
    normalized = _text(text)
    if not normalized:
        return []
    buffer: list[str] = []
    current = ""
    for char in normalized:
        current += char
        if char in _SPLIT_MARKERS:
            sentence = current.strip()
            if sentence:
                buffer.append(sentence)
            current = ""
    if current.strip():
        buffer.append(current.strip())
    return buffer


def _contains_any(text: str, patterns: Sequence[str]) -> bool:
    normalized = _text(text)
    return any(pattern in normalized for pattern in patterns)


def _contains_question(sentence: str) -> bool:
    text = _text(sentence)
    if not text:
        return False
    if "?" in text or "？" in text:
        return True
    return any(pattern in text for pattern in _QUESTION_PHRASES)


def _contains_uncertainty_meta(sentence: str) -> bool:
    return _contains_any(sentence, _UNCERTAINTY_META_PHRASES)


def _matches_any_pattern(
    text: str,
    patterns: Sequence[re.Pattern[str]],
) -> bool:
    normalized = _text(text)
    return any(pattern.search(normalized) for pattern in patterns)


@dataclass(frozen=True)
class _SentenceFunctionSignal:
    requests_information: bool = False
    interprets_meaning: bool = False
    heavy_surface_act: bool = False


def _sentence_function_signal(sentence: str) -> _SentenceFunctionSignal:
    normalized = _text(sentence)
    if not normalized:
        return _SentenceFunctionSignal()
    requests_information = (
        "?" in normalized
        or "？" in normalized
        or _matches_any_pattern(normalized, _INFORMATION_REQUEST_PATTERNS)
    )
    interprets_meaning = _matches_any_pattern(
        normalized,
        _INTERPRETATION_ACT_PATTERNS,
    )
    heavy_surface_act = _matches_any_pattern(
        normalized,
        _HEAVY_SURFACE_ACT_PATTERNS,
    )
    return _SentenceFunctionSignal(
        requests_information=requests_information,
        interprets_meaning=interprets_meaning,
        heavy_surface_act=heavy_surface_act,
    )


def _classify_small_shared_moment_sentence(sentence: str) -> set[str]:
    normalized = _text(sentence)
    if not normalized:
        return set()
    signal = _sentence_function_signal(normalized)
    classes: set[str] = set()
    if _contains_uncertainty_meta(normalized):
        classes.add("uncertainty_meta")
    if _contains_question(normalized) or signal.requests_information:
        classes.add("question")
    if _contains_any(normalized, _ELICITATION_PHRASES) or _matches_any_pattern(
        normalized,
        _ELICITATION_PATTERNS,
    ) or signal.heavy_surface_act:
        classes.add("elicitation")
    if _contains_any(normalized, _ASSISTANT_ATTRACTOR_PHRASES) or _matches_any_pattern(
        normalized,
        _ASSISTANT_ATTRACTOR_PATTERNS,
    ) or signal.heavy_surface_act:
        classes.add("assistant_attractor")
    if _contains_any(
        normalized,
        _INTERPRETIVE_BRIGHT_PHRASES,
    ) or _matches_any_pattern(
        normalized,
        _INTERPRETIVE_BRIGHT_PATTERNS,
    ) or signal.interprets_meaning:
        classes.add("interpretive_bright")
    if _matches_any_pattern(normalized, _SMALL_SHARED_REACTION_PATTERNS):
        classes.add("small_shared_reaction")
    return classes


def _is_small_shared_moment(
    surface_context_packet: Mapping[str, Any] | None,
    *,
    fallback_text: str = "",
) -> bool:
    source_state = _source_state(surface_context_packet)
    packet_surface_profile = (
        surface_context_packet.get("surface_profile")
        if isinstance(surface_context_packet, Mapping)
        and isinstance(surface_context_packet.get("surface_profile"), Mapping)
        else {}
    )
    conversation_phase = _text(
        (surface_context_packet or {}).get("conversation_phase")
        if isinstance(surface_context_packet, Mapping)
        else ""
    )
    offer_mode = _text(source_state.get("utterance_reason_offer"))
    preserve_mode = _text(source_state.get("utterance_reason_preserve"))
    shared_moment_kind = _text(source_state.get("shared_moment_kind"))
    turn_delta_kind = _text(source_state.get("turn_delta_kind"))
    discourse_shape_id = _text(source_state.get("discourse_shape_id"))
    if not discourse_shape_id:
        discourse_shape_id = _text(packet_surface_profile.get("discourse_shape_id"))
    if offer_mode == "brief_shared_smile":
        return True
    if preserve_mode == "keep_it_small":
        return True
    if conversation_phase == "bright_continuity":
        return True
    if turn_delta_kind == "bright_continuity":
        return True
    if discourse_shape_id == "bright_bounce":
        return True
    fallback = _text(fallback_text)
    if fallback:
        fallback_sentences = _split_sentences(fallback)
        fallback_is_small = len(fallback_sentences) <= 2
        fallback_is_reaction = _matches_any_pattern(
            fallback,
            _SMALL_SHARED_REACTION_PATTERNS,
        )
        if fallback_is_small and fallback_is_reaction:
            return True
    return (
        conversation_phase in {"continuing_thread", "reopening_thread"}
        and shared_moment_kind in {"laugh", "smile", "relief"}
    )


@dataclass(frozen=True)
class LLMBridgeContractViolation:
    code: str
    detail: str


@dataclass(frozen=True)
class LLMBridgeContractReview:
    ok: bool
    raw_text: str
    sanitized_text: str
    violations: tuple[LLMBridgeContractViolation, ...] = ()

    def violation_codes(self) -> list[str]:
        return [violation.code for violation in self.violations]


@dataclass(frozen=True)
class _LLMBridgeReviewContext:
    normalized_raw_text: str
    normalized_fallback_text: str
    question_policy: str
    interpretation_budget: str
    scale: str
    sentence_cap: int
    sentences: tuple[str, ...]
    sentence_classes: tuple[set[str], ...]
    sentence_signals: tuple[_SentenceFunctionSignal, ...]
    speech_act_analysis: SpeechActAnalysis | None = None


def _non_meta_sentence_count(context: _LLMBridgeReviewContext) -> int:
    return len(
        [
            classes
            for classes in context.sentence_classes
            if "uncertainty_meta" not in classes
        ]
    )


def _detect_common_violations(
    context: _LLMBridgeReviewContext,
) -> list[LLMBridgeContractViolation]:
    violations: list[LLMBridgeContractViolation] = []
    if _contains_any(context.normalized_raw_text, _UNCERTAINTY_META_PHRASES):
        violations.append(
            LLMBridgeContractViolation(
                code="uncertainty_meta_violation",
                detail="raw text に推定信頼度や不確実要因などの meta text が混ざっています。",
            )
        )
    if context.scale in {"micro", "small"} and any(
        signal.heavy_surface_act for signal in context.sentence_signals
    ):
        violations.append(
            LLMBridgeContractViolation(
                code="surface_scale_violation",
                detail="reaction_contract の scale に対して、発話行為が重くなっています。",
            )
        )
    return violations


def _detect_speech_act_violations(
    context: _LLMBridgeReviewContext,
    *,
    small_shared_moment: bool,
) -> list[LLMBridgeContractViolation]:
    analysis = context.speech_act_analysis
    if analysis is None:
        return []
    violations: list[LLMBridgeContractViolation] = []
    if context.question_policy == "none" and analysis.has_label("information_request"):
        violations.append(
            LLMBridgeContractViolation(
                code="question_block_violation",
                detail="speech_act classifier detected an information_request while question_policy=none.",
            )
        )
    if context.interpretation_budget == "none" and analysis.has_label("interpretation"):
        violations.append(
            LLMBridgeContractViolation(
                code="interpretation_budget_violation",
                detail="speech_act classifier detected interpretation while interpretation_budget=none.",
            )
        )
    if context.scale in {"micro", "small"} and analysis.has_label("advice_or_directive"):
        violations.append(
            LLMBridgeContractViolation(
                code="surface_scale_violation",
                detail="speech_act classifier detected advice_or_directive for a micro/small response.",
            )
        )
    if analysis.has_label("meta_commentary"):
        violations.append(
            LLMBridgeContractViolation(
                code="uncertainty_meta_violation",
                detail="speech_act classifier detected evaluator or uncertainty meta-commentary in the output.",
            )
        )
    if small_shared_moment and analysis.has_label("support_offer"):
        violations.append(
            LLMBridgeContractViolation(
                code="assistant_attractor_violation",
                detail="speech_act classifier detected support_offer in a small shared moment.",
            )
        )
    return violations


def _check_non_small_shared(
    context: _LLMBridgeReviewContext,
) -> list[LLMBridgeContractViolation]:
    violations: list[LLMBridgeContractViolation] = []
    if context.question_policy == "none" and any(
        "question" in classes for classes in context.sentence_classes
    ):
        violations.append(
            LLMBridgeContractViolation(
                code="question_block_violation",
                detail="reaction_contract では質問しない場面だが、follow-up question が含まれています。",
            )
        )
    if context.interpretation_budget == "none" and any(
        "interpretive_bright" in classes for classes in context.sentence_classes
    ):
        violations.append(
            LLMBridgeContractViolation(
                code="interpretation_budget_violation",
                detail="reaction_contract では解釈を足さない場面だが、解釈文が含まれています。",
            )
        )
    if context.sentence_cap and _non_meta_sentence_count(context) > context.sentence_cap:
        violations.append(
            LLMBridgeContractViolation(
                code="too_many_sentences",
                detail="reaction_contract の scale より文数が多すぎます。",
            )
        )
    return violations


def _check_small_shared(
    context: _LLMBridgeReviewContext,
) -> list[LLMBridgeContractViolation]:
    violations: list[LLMBridgeContractViolation] = []
    if context.question_policy == "none" and any(
        "question" in classes for classes in context.sentence_classes
    ):
        violations.append(
            LLMBridgeContractViolation(
                code="question_block_violation",
                detail="question_policy=none なのに follow-up question が入っています。",
            )
        )
    if any("elicitation" in classes for classes in context.sentence_classes):
        violations.append(
            LLMBridgeContractViolation(
                code="elicitation_violation",
                detail="小さい共有モーメントを広げる促し文が混ざっています。",
            )
        )
    if any("assistant_attractor" in classes for classes in context.sentence_classes):
        violations.append(
            LLMBridgeContractViolation(
                code="assistant_attractor_violation",
                detail="assistant/counselor phrasing に引っ張られています。",
            )
        )
    if any("interpretive_bright" in classes for classes in context.sentence_classes):
        violations.append(
            LLMBridgeContractViolation(
                code="interpretive_bright_violation",
                detail="小さい共有モーメントを解釈文へ広げています。",
            )
        )
    allowed_sentence_count = context.sentence_cap or 2
    if _non_meta_sentence_count(context) > allowed_sentence_count:
        violations.append(
            LLMBridgeContractViolation(
                code="too_many_sentences",
                detail="keep_it_small の場面で 3 文以上に広がっています。",
            )
        )
    return violations


def _sanitize_small_shared_text(context: _LLMBridgeReviewContext) -> str:
    sanitized_sentences: list[str] = []
    blocked_patterns: tuple[str, ...] = (
        *_ASSISTANT_ATTRACTOR_PHRASES,
        *_INTERPRETIVE_BRIGHT_PHRASES,
        *_ELICITATION_PHRASES,
        *_UNCERTAINTY_META_PHRASES,
    )
    blocked_sentence_classes = {
        "question",
        "elicitation",
        "assistant_attractor",
        "interpretive_bright",
        "uncertainty_meta",
    }
    for sentence, classes in zip(context.sentences, context.sentence_classes):
        if context.question_policy == "none" and "question" in classes:
            continue
        if classes.intersection(blocked_sentence_classes):
            continue
        if _contains_any(sentence, blocked_patterns):
            continue
        sanitized_sentences.append(sentence)

    allowed_sentence_count = context.sentence_cap or 2
    sanitized_sentences = sanitized_sentences[: max(1, allowed_sentence_count)]
    sanitized_text = " ".join(sentence.strip() for sentence in sanitized_sentences).strip()
    if context.normalized_fallback_text:
        return context.normalized_fallback_text
    return sanitized_text or context.normalized_raw_text


def review_llm_bridge_text(
    *,
    raw_text: str,
    surface_context_packet: Mapping[str, Any] | None = None,
    reaction_contract: Mapping[str, Any] | None = None,
    fallback_text: str = "",
    speech_act_analysis: Mapping[str, Any] | None = None,
) -> LLMBridgeContractReview:
    normalized_raw_text = _text(raw_text)
    normalized_fallback_text = _text(fallback_text)
    if not normalized_raw_text:
        return LLMBridgeContractReview(
            ok=True,
            raw_text="",
            sanitized_text=normalized_fallback_text,
            violations=(),
        )

    constraints = _constraints(surface_context_packet)
    source_state = _source_state(surface_context_packet)
    reaction = _reaction_contract(reaction_contract)
    question_policy = _text(source_state.get("utterance_reason_question_policy"))
    if not question_policy and int(constraints.get("max_questions") or 0) <= 0:
        question_policy = "none"
    if not question_policy and int(reaction.get("question_budget") or 0) <= 0:
        question_policy = "none"
    is_small_shared_moment = _is_small_shared_moment(
        surface_context_packet,
        fallback_text=normalized_fallback_text,
    )
    scale = _text(reaction.get("scale"))
    interpretation_budget = _text(reaction.get("interpretation_budget"))
    if not is_small_shared_moment and scale in {"micro", "small"}:
        is_small_shared_moment = True

    sentences = _split_sentences(normalized_raw_text)
    sentence_classes = [
        _classify_small_shared_moment_sentence(sentence)
        for sentence in sentences
    ]
    sentence_signals = [
        _sentence_function_signal(sentence)
        for sentence in sentences
    ]
    parsed_speech_act_analysis = (
        speech_act_analysis_from_dict(speech_act_analysis)
        if isinstance(speech_act_analysis, Mapping)
        else None
    )
    sentence_cap = 0
    if scale == "micro":
        sentence_cap = 1
    elif scale == "small":
        sentence_cap = 2
    elif scale == "medium":
        sentence_cap = 3

    context = _LLMBridgeReviewContext(
        normalized_raw_text=normalized_raw_text,
        normalized_fallback_text=normalized_fallback_text,
        question_policy=question_policy,
        interpretation_budget=interpretation_budget,
        scale=scale,
        sentence_cap=sentence_cap,
        sentences=tuple(sentences),
        sentence_classes=tuple(sentence_classes),
        sentence_signals=tuple(sentence_signals),
        speech_act_analysis=parsed_speech_act_analysis,
    )

    if not is_small_shared_moment:
        violations = [
            *_detect_speech_act_violations(
                context,
                small_shared_moment=False,
            ),
            *_detect_common_violations(context),
            *_check_non_small_shared(context),
        ]
        if violations:
            return LLMBridgeContractReview(
                ok=False,
                raw_text=normalized_raw_text,
                sanitized_text=normalized_fallback_text or normalized_raw_text,
                violations=tuple(violations),
            )
        return LLMBridgeContractReview(
            ok=True,
            raw_text=normalized_raw_text,
            sanitized_text=normalized_raw_text,
            violations=(),
        )

    violations = [
        *_detect_speech_act_violations(
            context,
            small_shared_moment=True,
        ),
        *_detect_common_violations(context),
        *_check_small_shared(context),
    ]

    if not violations:
        return LLMBridgeContractReview(
            ok=True,
            raw_text=normalized_raw_text,
            sanitized_text=normalized_raw_text,
            violations=(),
        )

    return LLMBridgeContractReview(
        ok=False,
        raw_text=normalized_raw_text,
        sanitized_text=_sanitize_small_shared_text(context),
        violations=tuple(violations),
    )
