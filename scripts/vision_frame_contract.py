from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"1", "true", "yes", "on"}:
            return True
        if raw in {"0", "false", "no", "off"}:
            return False
    return default


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


SENSOR_FRAME_KEYS = {
    "pose_vec",
    "voice_level",
    "breath_rate",
    "inner_emotion_score",
    "heart_rate_raw",
    "heart_rate_baseline",
    "body_stress_index",
    "autonomic_balance",
    "object_counts",
    "pose_detected",
    "person_count",
    "motion_score",
    "has_face",
    "place_id",
    "privacy_tags",
    "vlm_features",
    "audio",
    "body",
    "place",
}


def build_sensor_frame(payload: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    sensor_payload = payload.get("sensor")
    if not isinstance(sensor_payload, Mapping):
        sensor_payload = payload.get("sensor_frame")
    if not isinstance(sensor_payload, Mapping):
        if any(key in payload for key in SENSOR_FRAME_KEYS):
            sensor_payload = payload
        else:
            return None

    raw: Dict[str, Any] = {}
    _merge_direct_fields(raw, sensor_payload)
    _merge_audio_channel(raw, sensor_payload.get("audio"))
    _merge_body_channel(raw, sensor_payload.get("body"))
    _merge_place_channel(raw, sensor_payload.get("place"))

    detections = sensor_payload.get("detections")
    if isinstance(detections, list):
        _merge_detection_hints(raw, detections)

    if not raw:
        return None
    raw.setdefault("has_face", False)
    raw.setdefault("pose_detected", bool(raw.get("pose_vec")))
    raw.setdefault("person_count", 0)
    raw.setdefault("object_counts", {})
    return raw


def summarize_sensor_frame(raw_frame: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not isinstance(raw_frame, Mapping):
        return {"available": False}
    return {
        "available": True,
        "person_count": int(raw_frame.get("person_count", 0) or 0),
        "has_face": bool(raw_frame.get("has_face", False)),
        "motion_score": round(_coerce_float(raw_frame.get("motion_score"), 0.0), 4),
        "voice_level": round(_coerce_float(raw_frame.get("voice_level"), 0.0), 4),
        "breath_rate": round(_coerce_float(raw_frame.get("breath_rate"), 0.0), 4),
        "body_stress_index": round(_coerce_float(raw_frame.get("body_stress_index"), 0.0), 4),
        "autonomic_balance": round(_coerce_float(raw_frame.get("autonomic_balance"), 0.0), 4),
        "object_keys": sorted((raw_frame.get("object_counts") or {}).keys()),
        "place_id": str(raw_frame.get("place_id") or ""),
        "privacy_tags": list(raw_frame.get("privacy_tags") or []),
    }


def _merge_direct_fields(raw: Dict[str, Any], sensor_payload: Mapping[str, Any]) -> None:
    if "pose_vec" in sensor_payload and isinstance(sensor_payload.get("pose_vec"), Iterable):
        raw["pose_vec"] = [
            _coerce_float(value) for value in sensor_payload.get("pose_vec") if isinstance(value, (int, float, str))
        ]
    if "vlm_features" in sensor_payload and isinstance(sensor_payload.get("vlm_features"), Iterable):
        raw["vlm_features"] = [
            _coerce_float(value) for value in sensor_payload.get("vlm_features") if isinstance(value, (int, float, str))
        ]

    for key in (
        "voice_level",
        "breath_rate",
        "inner_emotion_score",
        "heart_rate_raw",
        "heart_rate_baseline",
        "body_stress_index",
        "autonomic_balance",
        "motion_score",
    ):
        if key in sensor_payload:
            raw[key] = _coerce_float(sensor_payload.get(key))

    if "person_count" in sensor_payload:
        raw["person_count"] = int(_coerce_float(sensor_payload.get("person_count")))

    for key in ("has_face", "pose_detected"):
        if key in sensor_payload:
            raw[key] = _coerce_bool(sensor_payload.get(key))

    object_counts = sensor_payload.get("object_counts")
    if isinstance(object_counts, Mapping):
        raw["object_counts"] = {
            str(name): int(_coerce_float(count))
            for name, count in object_counts.items()
            if str(name).strip()
        }

    place_id = sensor_payload.get("place_id")
    if place_id is not None:
        raw["place_id"] = str(place_id)

    privacy_tags = sensor_payload.get("privacy_tags")
    if isinstance(privacy_tags, Iterable) and not isinstance(privacy_tags, (str, bytes, Mapping)):
        raw["privacy_tags"] = [str(tag) for tag in privacy_tags if str(tag).strip()]


def _merge_audio_channel(raw: Dict[str, Any], audio_payload: Any) -> None:
    if not isinstance(audio_payload, Mapping):
        return
    if "voice_level" in audio_payload:
        raw["voice_level"] = _coerce_float(audio_payload.get("voice_level"))
    if "breath_rate" in audio_payload:
        raw["breath_rate"] = _coerce_float(audio_payload.get("breath_rate"))
    if "speaking" in audio_payload and "has_face" not in raw:
        raw["has_face"] = _coerce_bool(audio_payload.get("speaking"), raw.get("has_face", False))


def _merge_body_channel(raw: Dict[str, Any], body_payload: Any) -> None:
    if not isinstance(body_payload, Mapping):
        return
    for key in (
        "heart_rate_raw",
        "heart_rate_baseline",
        "body_stress_index",
        "autonomic_balance",
        "inner_emotion_score",
        "motion_score",
    ):
        if key in body_payload:
            raw[key] = _coerce_float(body_payload.get(key))
    if "pose_vec" in body_payload and isinstance(body_payload.get("pose_vec"), Iterable):
        raw["pose_vec"] = [
            _coerce_float(value) for value in body_payload.get("pose_vec") if isinstance(value, (int, float, str))
        ]
    if "pose_detected" in body_payload:
        raw["pose_detected"] = _coerce_bool(body_payload.get("pose_detected"))


def _merge_place_channel(raw: Dict[str, Any], place_payload: Any) -> None:
    if not isinstance(place_payload, Mapping):
        return
    if "place_id" in place_payload:
        raw["place_id"] = str(place_payload.get("place_id") or "")
    privacy_tags = place_payload.get("privacy_tags")
    if isinstance(privacy_tags, Iterable) and not isinstance(privacy_tags, (str, bytes, Mapping)):
        raw["privacy_tags"] = [str(tag) for tag in privacy_tags if str(tag).strip()]


def _merge_detection_hints(raw: Dict[str, Any], detections: list[Mapping[str, Any]]) -> None:
    object_counts = dict(raw.get("object_counts") or {})
    person_count = int(raw.get("person_count", 0) or 0)
    pose_detected = bool(raw.get("pose_detected", False))

    for detection in detections:
        label = str(detection.get("class") or detection.get("label") or "").strip().lower()
        if not label:
            continue
        object_counts[label] = int(object_counts.get(label, 0)) + 1
        if label == "person":
            person_count += 1
        if detection.get("pose"):
            pose_detected = True

    if object_counts:
        raw["object_counts"] = object_counts
    if person_count > 0:
        raw["person_count"] = person_count
        raw.setdefault("has_face", True)
    raw["pose_detected"] = pose_detected
