from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


SPEECH_ACT_SCHEMA_VERSION = "speech_act.v1"

SPEECH_ACT_LABELS: tuple[str, ...] = (
    "acknowledgement",
    "information_request",
    "interpretation",
    "advice_or_directive",
    "meta_commentary",
    "support_offer",
    "small_shared_reaction",
    "other",
)


@dataclass(frozen=True)
class SpeechActSentence:
    text: str
    labels: tuple[str, ...]
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "labels": list(self.labels),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class SpeechActAnalysis:
    sentences: tuple[SpeechActSentence, ...]
    source: str = "external_classifier"
    schema_version: str = SPEECH_ACT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "sentences": [sentence.to_dict() for sentence in self.sentences],
        }

    def has_label(self, label: str) -> bool:
        return any(label in sentence.labels for sentence in self.sentences)


def build_speech_act_classification_request(text: str) -> dict[str, str]:
    labels = ", ".join(SPEECH_ACT_LABELS)
    return {
        "system_prompt": (
            "You are a speech-act classifier for a contract evaluator. "
            "Return JSON only. Do not rewrite or answer the user text."
        ),
        "user_prompt": (
            "Classify each sentence in the assistant output.\n"
            f"schema_version: {SPEECH_ACT_SCHEMA_VERSION}\n"
            f"allowed_labels: {labels}\n"
            "Return this JSON shape:\n"
            "{"
            '"schema_version":"speech_act.v1",'
            '"source":"external_classifier",'
            '"sentences":[{"text":"...","labels":["..."],"confidence":0.0}]'
            "}\n"
            "Use information_request for any request for more detail, even without a question mark.\n"
            "Use interpretation for inferred meaning, cause, emotional reading, or explanatory framing.\n"
            "Use advice_or_directive for suggested actions, instructions, or guidance.\n"
            "Use meta_commentary for confidence, uncertainty, policy, or evaluator explanations.\n"
            "Use small_shared_reaction for brief shared laughter, relief, or light acknowledgement.\n\n"
            f"assistant_output:\n{text}"
        ),
    }


def speech_act_analysis_from_dict(payload: Mapping[str, Any]) -> SpeechActAnalysis:
    raw_sentences = payload.get("sentences")
    if not isinstance(raw_sentences, Sequence) or isinstance(raw_sentences, (str, bytes)):
        raw_sentences = ()
    sentences: list[SpeechActSentence] = []
    allowed = set(SPEECH_ACT_LABELS)
    for raw_sentence in raw_sentences:
        if not isinstance(raw_sentence, Mapping):
            continue
        text = str(raw_sentence.get("text") or "").strip()
        raw_labels = raw_sentence.get("labels")
        labels: list[str] = []
        if isinstance(raw_labels, Sequence) and not isinstance(raw_labels, (str, bytes)):
            labels = [
                str(label).strip()
                for label in raw_labels
                if str(label).strip() in allowed
            ]
        confidence = _coerce_confidence(raw_sentence.get("confidence"))
        sentences.append(
            SpeechActSentence(
                text=text,
                labels=tuple(dict.fromkeys(labels or ["other"])),
                confidence=confidence,
            )
        )
    schema_version = str(payload.get("schema_version") or SPEECH_ACT_SCHEMA_VERSION)
    source = str(payload.get("source") or "external_classifier")
    return SpeechActAnalysis(
        sentences=tuple(sentences),
        source=source,
        schema_version=schema_version,
    )


def _coerce_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))
