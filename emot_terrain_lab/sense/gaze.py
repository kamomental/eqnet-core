# -*- coding: utf-8 -*-
"""Gaze-LLE helpers to project fixations into SenseEnvelope + summaries."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

from .envelope import SenseEnvelope, clamp_features


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _norm(value: float, max_value: float) -> float:
    if max_value <= 0:
        return 0.0
    return _clamp_unit(value / max_value)


@dataclass
class GazeSummary:
    """Compact summary of the most recent fixation."""

    target_id: Optional[str]
    label: Optional[str]
    fixation_ms: float
    saccade_rate_hz: float
    blink_rate_hz: float
    pupil_z: float
    mutual_gaze: float
    gaze_on_me: float
    cone_width_deg: float
    confidence: float

    def to_dict(self) -> Dict[str, float | str | None]:
        return {
            "target_id": self.target_id,
            "label": self.label,
            "fixation_ms": float(self.fixation_ms),
            "saccade_rate_hz": float(self.saccade_rate_hz),
            "blink_rate_hz": float(self.blink_rate_hz),
            "pupil_z": float(self.pupil_z),
            "mutual_gaze": float(self.mutual_gaze),
            "gaze_on_me": float(self.gaze_on_me),
            "cone_width_deg": float(self.cone_width_deg),
            "confidence": float(self.confidence),
        }


def extract_gaze_features(
    payload: Mapping[str, object] | None,
    cfg: Mapping[str, object] | None,
    *,
    t_tau: float,
) -> Optional[Dict[str, object]]:
    """Return SenseEnvelope-compatible gaze features plus summary."""

    if not isinstance(payload, Mapping):
        return None
    cfg = cfg or {}
    fixation = payload.get("fixation")
    if isinstance(fixation, Mapping):
        target_id = fixation.get("target_id") or fixation.get("object_id")
        fixation_ms = _safe(fixation.get("duration_ms"), _safe(payload.get("fix_dur_ms"), 0.0))
        confidence = _safe(
            fixation.get("confidence"),
            _safe(payload.get("confidence"), cfg.get("default_confidence", 0.7)),
        )
    else:
        target_id = payload.get("target_id")
        fixation_ms = _safe(payload.get("fix_dur_ms"))
        confidence = _safe(payload.get("confidence"), cfg.get("default_confidence", 0.7))

    saccade_rate = _safe(payload.get("saccade_rate_hz"), cfg.get("saccade_baseline_hz", 2.0))
    blink_rate = _safe(payload.get("blink_rate_hz"), cfg.get("blink_baseline_hz", 1.0))
    pupil_z = _safe(payload.get("pupil_z"))
    mutual_gaze = _clamp_unit(_safe(payload.get("mutual_gaze")))
    gaze_on_me = _clamp_unit(_safe(payload.get("gaze_on_me")))
    cone_width = _safe(payload.get("cone_width_deg"), cfg.get("cone_baseline_deg", 5.0))

    label = None
    if isinstance(fixation, Mapping):
        label = fixation.get("label")
    if not label:
        label = payload.get("label")

    summary = GazeSummary(
        target_id=str(target_id) if target_id else None,
        label=str(label) if label else None,
        fixation_ms=fixation_ms,
        saccade_rate_hz=saccade_rate,
        blink_rate_hz=blink_rate,
        pupil_z=pupil_z,
        mutual_gaze=mutual_gaze,
        gaze_on_me=gaze_on_me,
        cone_width_deg=cone_width,
        confidence=confidence,
    )

    max_fix_ms = _safe(cfg.get("max_fix_ms"), 1200.0)
    max_saccade = _safe(cfg.get("max_saccade_hz"), 5.0)
    max_blink = _safe(cfg.get("max_blink_hz"), 3.0)
    max_cone = _safe(cfg.get("max_cone_deg"), 15.0)
    pupil_scale = _safe(cfg.get("pupil_scale"), 3.0)

    features = clamp_features(
        {
            "fixation_strength": _norm(fixation_ms, max_fix_ms),
            "mutual_gaze": mutual_gaze,
            "gaze_on_me": gaze_on_me,
            "saccade_rate": _norm(saccade_rate, max_saccade),
            "blink_rate": _norm(blink_rate, max_blink),
            "pupil_arousal": _clamp_unit(0.5 + 0.5 * math.tanh(pupil_z / max(pupil_scale, 1e-6))),
            "cone_width": _norm(cone_width, max_cone),
        }
    )

    tags = []
    if summary.target_id:
        tags.append(f"obj:{summary.target_id}")

    envelope = SenseEnvelope(
        id=str(payload.get("id") or uuid.uuid4().hex),
        modality="vision_gaze",
        features=features,
        confidence=confidence,
        source=str(payload.get("source") or "external"),
        t_tau=float(payload.get("t_tau") or t_tau),
        tags=tags,
    )

    return {
        "envelope": envelope,
        "summary": summary,
    }


__all__ = ["GazeSummary", "extract_gaze_features"]
