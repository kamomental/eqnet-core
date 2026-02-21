"""Temporal state vector primitives for retrieval and update gating."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


@lru_cache(maxsize=4)
def status_lexicon(locale: str = "ja") -> Dict[str, str]:
    normalized = (locale or "ja").lower()
    if not normalized.startswith("ja"):
        return {
            "focus_with_tags": "Current focus: {tags}",
            "focus_balanced": "Current focus: balanced",
            "unresolved_high": "open loops are elevated",
            "unresolved_mid": "open loops remain",
            "unresolved_low": "open loops are manageable",
            "stability_unknown": "stability: unknown",
            "stability_stable": "stability: stable",
            "stability_shifting": "stability: shifting",
            "stability_change_point": "stability: near change point",
            "sat_high": "retrieval saturation is high.",
        }
    defaults = {
        "focus_with_tags": "Current focus: {tags}",
        "focus_balanced": "Current focus: balanced",
        "unresolved_high": "open loops are elevated",
        "unresolved_mid": "open loops remain",
        "unresolved_low": "open loops are manageable",
        "stability_unknown": "stability: unknown",
        "stability_stable": "stability: stable",
        "stability_shifting": "stability: shifting",
        "stability_change_point": "stability: near change point",
        "sat_high": "retrieval saturation is high.",
    }
    path = Path(__file__).resolve().parents[2] / "locales" / "ja.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        section = payload.get("state_vector_status")
        if isinstance(section, dict):
            out = dict(defaults)
            for key in defaults:
                value = section.get(key)
                if isinstance(value, str) and value.strip():
                    out[key] = value
            return out
    except Exception:
        pass
    return defaults


@dataclass(slots=True)
class TemporalStateVector:
    """Minimal, serializable state used by associative retrieval."""

    timestamp_ms: int
    valence: float = 0.0
    arousal: float = 0.0
    value_tags: Dict[str, float] = field(default_factory=dict)
    open_loops: float = 0.0
    event_scale: float = 0.0
    embed: Sequence[float] = field(default_factory=tuple)
    version: str = "tsv1"

    def to_status_text(
        self,
        *,
        coherence: float | None = None,
        sat_ratio: float | None = None,
        max_tags: int = 3,
        locale: str = "ja",
    ) -> str:
        tags = sorted(self.value_tags.items(), key=lambda kv: kv[1], reverse=True)
        top = [k for k, v in tags if v > 0.0][: max(1, int(max_tags))]
        lex = status_lexicon(locale)

        if locale.lower().startswith("ja"):
            joined = "」「".join(top)
            focus = (
                str(lex["focus_with_tags"]).format(tags=joined)
                if top
                else str(lex["focus_balanced"])
            )
            if self.open_loops >= 0.6:
                unresolved = str(lex["unresolved_high"])
            elif self.open_loops >= 0.2:
                unresolved = str(lex["unresolved_mid"])
            else:
                unresolved = str(lex["unresolved_low"])

            if coherence is None:
                stability = str(lex["stability_unknown"])
            elif coherence >= 0.7:
                stability = str(lex["stability_stable"])
            elif coherence >= 0.4:
                stability = str(lex["stability_shifting"])
            else:
                stability = str(lex["stability_change_point"])

            sat_note = ""
            if sat_ratio is not None and sat_ratio >= 0.6:
                sat_note = " " + str(lex["sat_high"])
            return f"{focus}。{unresolved}、{stability}。{sat_note}".replace("。。", "。").strip()

        focus = (
            str(lex["focus_with_tags"]).format(tags=", ".join(top))
            if top
            else str(lex["focus_balanced"])
        )
        if self.open_loops >= 0.6:
            unresolved = str(lex["unresolved_high"])
        elif self.open_loops >= 0.2:
            unresolved = str(lex["unresolved_mid"])
        else:
            unresolved = str(lex["unresolved_low"])

        if coherence is None:
            stability = str(lex["stability_unknown"])
        elif coherence >= 0.7:
            stability = str(lex["stability_stable"])
        elif coherence >= 0.4:
            stability = str(lex["stability_shifting"])
        else:
            stability = str(lex["stability_change_point"])

        sat_note = ""
        if sat_ratio is not None and sat_ratio >= 0.6:
            sat_note = "; " + str(lex["sat_high"])
        return f"{focus}. {unresolved}, {stability}{sat_note}."

    def to_assoc_context(self, *, temporal_tau_sec: float | None = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "timestamp_ms": int(self.timestamp_ms),
            "valence": float(self.valence),
            "arousal": float(self.arousal),
            "value_tags": dict(self.value_tags),
            "open_loops": float(self.open_loops),
            "event_scale": float(self.event_scale),
        }
        if temporal_tau_sec is not None:
            payload["temporal_tau_sec"] = float(temporal_tau_sec)
        return payload

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_ms": int(self.timestamp_ms),
            "valence": float(self.valence),
            "arousal": float(self.arousal),
            "value_tags": dict(self.value_tags),
            "open_loops": float(self.open_loops),
            "event_scale": float(self.event_scale),
            "embed": list(self.embed),
            "version": self.version,
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "TemporalStateVector":
        raw_tags = payload.get("value_tags")
        tags: Dict[str, float] = {}
        if isinstance(raw_tags, Mapping):
            for key, value in raw_tags.items():
                label = str(key).strip()
                if not label:
                    continue
                try:
                    tags[label] = float(value)
                except (TypeError, ValueError):
                    continue
        embed_raw = payload.get("embed")
        embed = tuple(embed_raw) if isinstance(embed_raw, (list, tuple)) else tuple()
        return TemporalStateVector(
            timestamp_ms=int(payload.get("timestamp_ms", 0)),
            valence=_safe_float(payload.get("valence"), 0.0),
            arousal=_safe_float(payload.get("arousal"), 0.0),
            value_tags=tags,
            open_loops=_safe_float(payload.get("open_loops"), 0.0),
            event_scale=_safe_float(payload.get("event_scale"), 0.0),
            embed=embed,
            version=str(payload.get("version", "tsv1")),
        )


def temporal_delta(prev: TemporalStateVector, curr: TemporalStateVector) -> float:
    """Simple delta used as an input signal for coherence gating."""
    base = (
        abs(curr.valence - prev.valence)
        + abs(curr.arousal - prev.arousal)
        + abs(curr.open_loops - prev.open_loops)
        + abs(curr.event_scale - prev.event_scale)
    )
    return float(base / 4.0)


def coherence_score(prev: TemporalStateVector, curr: TemporalStateVector) -> float:
    """Context-scaled coherence in [0, 1]. Higher is more coherent."""
    delta = temporal_delta(prev, curr)
    expected = max(0.05, min(1.0, abs(curr.event_scale)))
    ratio = delta / expected
    return float(max(0.0, min(1.0, 1.0 - min(1.0, ratio))))


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)

