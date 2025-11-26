"""Culture climate + monument model for EQNet."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Callable, DefaultDict, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

from collections import defaultdict
import math
import re
import time


ContextPayload = Union["CultureContext", Mapping[str, Optional[str]], None]
EventPayload = Mapping[str, object]

DEFAULT_CULTURE_TAG = "default"
DEFAULT_PLACE_ID = "unknown_place"
DEFAULT_PARTNER_ID = "solo"
DEFAULT_ACTIVITY_TAG = None
DEFAULT_OBJECT_ID = None
DEFAULT_OBJECT_ROLE = None

CLIMATE_HALF_LIFE_HOURS = 72.0
DEFAULT_CLIMATE_ALPHA = 0.3
DEFAULT_MONUMENT_THRESHOLD = 0.6
MAX_SPIKE_WEIGHT = 0.9
MIN_SPIKE_WEIGHT = 0.0

KEY_PHRASES = (
    r"忘れられない",
    r"今でも(覚えて|忘れない)",
    r"人生で一番",
    r"ターニングポイント",
    r"転機",
    r"あの日",
    r"あのとき",
    r"一生(の|モノ)",
    r"二度と(ない|来ない)",
)
INWARD_PHRASES = (
    r"一人で",
    r"誰にも話して",
    r"自分の中で",
    r"静かに泣いた",
)
FICTION_PHRASES = (
    r"この(アニメ|作品)",
    r"キャラ",
    r"物語",
)
FICTION_OBJECT_PREFIXES = ("anime:", "movie:", "game:", "novel:", "manga:")


@dataclass
class CultureContext:
    """Culture-related coordinates attached to each moment."""

    culture_tag: Optional[str] = None
    place_id: Optional[str] = None
    partner_id: Optional[str] = None
    object_id: Optional[str] = None
    object_role: Optional[str] = None
    activity_tag: Optional[str] = None

    def normalized(self) -> "CultureContext":
        return CultureContext(
            culture_tag=self.culture_tag or DEFAULT_CULTURE_TAG,
            place_id=self.place_id or DEFAULT_PLACE_ID,
            partner_id=self.partner_id or DEFAULT_PARTNER_ID,
            object_id=self.object_id or DEFAULT_OBJECT_ID,
            object_role=self.object_role or DEFAULT_OBJECT_ROLE,
            activity_tag=self.activity_tag or DEFAULT_ACTIVITY_TAG,
        )


@dataclass
class ClimateState:
    """EMA for culture, place, or partner buckets."""

    valence: float = 0.0
    arousal: float = 0.0
    intimacy: float = 0.0
    politeness: float = 0.0
    rho: float = 0.0
    n: int = 0
    last_ts: float = 0.0


class MonumentKind(str, Enum):
    SOCIAL = "social"
    PERSONAL = "personal"
    FICTION = "fiction"


@dataclass
class MemoryMonument:
    """Salient episodic landmark."""

    id: str
    kind: MonumentKind
    place_id: Optional[str]
    partner_id: Optional[str]
    culture_tag: Optional[str]
    object_id: Optional[str]
    object_role: Optional[str]
    valence: float
    arousal: float
    intimacy: float
    politeness: float
    rho: float
    salience: float
    created_ts: float
    replay_count: int = 0


@dataclass
class CultureState:
    valence: float = 0.0
    arousal: float = 0.0
    intimacy: float = 0.0
    politeness: float = 0.0
    rho: float = 0.0


@dataclass
class BehaviorMod:
    tone: str
    empathy_level: float
    directness: float
    joke_ratio: float


class CultureFieldStorage:
    """Holds EMA states per bucket."""

    def __init__(self) -> None:
        self.culture: DefaultDict[str, ClimateState] = defaultdict(ClimateState)
        self.place: DefaultDict[str, ClimateState] = defaultdict(ClimateState)
        self.partner: DefaultDict[str, ClimateState] = defaultdict(ClimateState)

    def bucket(self, name: str) -> DefaultDict[str, ClimateState]:
        return getattr(self, name)


class MonumentStorage:
    """In-memory store for monuments."""

    def __init__(self) -> None:
        self._items: Dict[str, MemoryMonument] = {}
        self._counter: int = 0

    def __len__(self) -> int:
        return len(self._items)

    def values(self) -> Iterable[MemoryMonument]:
        return self._items.values()

    def clear(self) -> None:
        self._items.clear()
        self._counter = 0

    def add(self, monument: MemoryMonument) -> None:
        self._items[monument.id] = monument

    def generate_id(self) -> str:
        self._counter += 1
        return f"mon_{int(time.time())}_{self._counter}"


_CLIMATE_STORAGE = CultureFieldStorage()
_MONUMENT_STORAGE = MonumentStorage()


def use_climate_storage(storage: CultureFieldStorage) -> None:
    global _CLIMATE_STORAGE
    _CLIMATE_STORAGE = storage


def use_monument_storage(storage: MonumentStorage) -> None:
    global _MONUMENT_STORAGE
    _MONUMENT_STORAGE = storage


def _as_context(payload: ContextPayload, event: Optional[EventPayload] = None) -> CultureContext:
    if isinstance(payload, CultureContext):
        ctx = payload
    else:
        data: Dict[str, Optional[str]] = {}
        if isinstance(payload, Mapping):
            for key in ("culture_tag", "place_id", "partner_id", "object_id", "object_role", "activity_tag"):
                val = payload.get(key)
                data[key] = str(val) if val is not None else None
        if isinstance(event, Mapping):
            for key in ("culture_tag", "place_id", "partner_id", "object_id", "object_role", "activity_tag"):
                if key not in data or data[key] is None:
                    val = event.get(key)
                    data[key] = str(val) if val not in (None, "") else None
        ctx = CultureContext(**data)
    return ctx.normalized()


def _decay_state(state: ClimateState, now_ts: float, half_life_hours: float) -> None:
    if state.n == 0 or state.last_ts == 0.0:
        return
    dt_hours = max((now_ts - state.last_ts) / 3600.0, 0.0)
    if dt_hours <= 0.0:
        return
    decay = math.pow(0.5, dt_hours / max(half_life_hours, 1e-6))
    state.valence *= decay
    state.arousal *= decay
    state.intimacy *= decay
    state.politeness *= decay
    state.rho *= decay


def _event_value(event: EventPayload, key: str, default: float = 0.0) -> float:
    value = event.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def update_climate_from_event(
    event: EventPayload,
    context: ContextPayload = None,
    *,
    alpha: float = DEFAULT_CLIMATE_ALPHA,
    half_life_hours: float = CLIMATE_HALF_LIFE_HOURS,
    storage: Optional[CultureFieldStorage] = None,
    timestamp: Optional[float] = None,
) -> None:
    """Update EMA states from a single moment."""

    ctx = _as_context(context, event)
    now_ts = timestamp or float(event.get("ts", time.time()))
    store = storage or _CLIMATE_STORAGE

    tags = (
        ("culture", ctx.culture_tag),
        ("place", ctx.place_id),
        ("partner", ctx.partner_id),
    )
    for bucket, tag in tags:
        if not tag:
            continue
        state = store.bucket(bucket)[tag]
        _decay_state(state, now_ts, half_life_hours)
        state.valence = (1 - alpha) * state.valence + alpha * _event_value(event, "valence")
        state.arousal = (1 - alpha) * state.arousal + alpha * _event_value(event, "arousal")
        state.intimacy = (1 - alpha) * state.intimacy + alpha * _event_value(event, "intimacy")
        state.politeness = (1 - alpha) * state.politeness + alpha * _event_value(event, "politeness")
        state.rho = (1 - alpha) * state.rho + alpha * _event_value(event, "rho")
        state.n += 1
        state.last_ts = now_ts


def extract_text_signal_score(text: str) -> float:
    if not text:
        return 0.0
    score = 0.0
    for pattern in KEY_PHRASES:
        if re.search(pattern, text):
            score += 0.2
    return min(score, 1.0)


def infer_monument_kind(event: EventPayload, text: str) -> MonumentKind:
    object_id = str(event.get("object_id")) if event.get("object_id") else ""
    partner_id = event.get("partner_id")
    text = text or ""
    if partner_id:
        return MonumentKind.SOCIAL
    if object_id.startswith(FICTION_OBJECT_PREFIXES) or any(re.search(pat, text) for pat in FICTION_PHRASES):
        return MonumentKind.FICTION
    if any(re.search(pat, text) for pat in INWARD_PHRASES):
        return MonumentKind.PERSONAL
    return MonumentKind.PERSONAL


def compute_monument_score(
    event: EventPayload,
    text: str,
    recurrence_count: int,
    *,
    explicit_pin: bool = False,
) -> float:
    val = abs(_event_value(event, "valence"))
    aro = abs(_event_value(event, "arousal"))
    emotion_strength = min((val + aro) / 2.0, 1.0)
    text_signal = extract_text_signal_score(text)
    rec_signal = min(max(recurrence_count, 0) / 5.0, 1.0)
    score = 0.5 * emotion_strength + 0.3 * text_signal + 0.2 * rec_signal
    if explicit_pin:
        score = max(score, 0.9)
    return min(score, 1.0)


def promote_to_monument_if_needed(
    event: EventPayload,
    text: str,
    recurrence_count: int,
    *,
    threshold: float = DEFAULT_MONUMENT_THRESHOLD,
    explicit_pin: bool = False,
    context: ContextPayload = None,
    storage: Optional[MonumentStorage] = None,
) -> Optional[MemoryMonument]:
    ctx = _as_context(context, event)
    anchors = [ctx.place_id, ctx.partner_id, ctx.object_id]
    if not any(a for a in anchors if a and a not in {DEFAULT_PLACE_ID, DEFAULT_PARTNER_ID, None}):
        return None
    store = storage or _MONUMENT_STORAGE
    score = compute_monument_score(event, text, recurrence_count, explicit_pin=explicit_pin)
    if score < threshold:
        return None
    kind = infer_monument_kind({**asdict(ctx), **event}, text)
    now_ts = float(event.get("ts", time.time()))
    existing = _find_similar_monument(store, ctx, kind)
    if existing:
        existing.replay_count += 1
        existing.salience = min(1.0, existing.salience + 0.1 * score)
        return existing
    monument = MemoryMonument(
        id=store.generate_id(),
        kind=kind,
        place_id=ctx.place_id,
        partner_id=ctx.partner_id,
        culture_tag=ctx.culture_tag,
        object_id=ctx.object_id,
        object_role=ctx.object_role,
        valence=_event_value(event, "valence"),
        arousal=_event_value(event, "arousal"),
        intimacy=_event_value(event, "intimacy"),
        politeness=_event_value(event, "politeness"),
        rho=_event_value(event, "rho"),
        salience=score,
        created_ts=now_ts,
        replay_count=1,
    )
    store.add(monument)
    return monument


def _find_similar_monument(
    store: MonumentStorage,
    ctx: CultureContext,
    kind: MonumentKind,
) -> Optional[MemoryMonument]:
    for monument in store.values():
        if (
            monument.kind == kind
            and monument.place_id == ctx.place_id
            and monument.partner_id == ctx.partner_id
            and monument.culture_tag == ctx.culture_tag
            and monument.object_id == ctx.object_id
        ):
            return monument
    return None


def rescan_events_for_monuments(
    events: Iterable[EventPayload],
    *,
    text_getter: Optional[Callable[[EventPayload], str]] = None,
    recurrence_getter: Optional[Callable[[EventPayload], int]] = None,
    storage: Optional[MonumentStorage] = None,
    reset_existing: bool = False,
) -> int:
    store = storage or _MONUMENT_STORAGE
    if reset_existing:
        store.clear()
    created = 0
    for event in events:
        text = text_getter(event) if text_getter else str(event.get("text", ""))
        recurrence = recurrence_getter(event) if recurrence_getter else int(event.get("recurrence", 0) or 0)
        if promote_to_monument_if_needed(event, text, recurrence, storage=store):
            created += 1
    return created


def _avg(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _avg3(a: float, b: float, c: float) -> float:
    return (a + b + c) / 3.0


def _get_state(tag: Optional[str], bucket: DefaultDict[str, ClimateState]) -> ClimateState:
    if tag and tag in bucket:
        return bucket[tag]
    return ClimateState()


def query_matching_monuments(
    *,
    context: ContextPayload,
    top_k: int = 3,
    storage: Optional[MonumentStorage] = None,
) -> List[MemoryMonument]:
    ctx = _as_context(context)
    store = storage or _MONUMENT_STORAGE
    candidates: List[Tuple[float, MemoryMonument]] = []
    for monument in store.values():
        score = 0.0
        if monument.place_id and monument.place_id == ctx.place_id:
            score += 0.4
        if monument.partner_id and monument.partner_id == ctx.partner_id:
            score += 0.3
        if monument.culture_tag and monument.culture_tag == ctx.culture_tag:
            score += 0.2
        if monument.kind == MonumentKind.FICTION and ctx.object_id and monument.object_id == ctx.object_id:
            score += 0.3
        score *= monument.salience
        if score > 0:
            candidates.append((score, monument))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [m for _, m in candidates[:top_k]]


def aggregate_monuments(monuments: Sequence[MemoryMonument]) -> CultureState:
    if not monuments:
        return CultureState()
    total_w = 0.0
    acc = {"valence": 0.0, "arousal": 0.0, "intimacy": 0.0, "politeness": 0.0, "rho": 0.0}
    for mon in monuments:
        weight = mon.salience * (1.0 + 0.2 * mon.replay_count)
        total_w += weight
        acc["valence"] += weight * mon.valence
        acc["arousal"] += weight * mon.arousal
        acc["intimacy"] += weight * mon.intimacy
        acc["politeness"] += weight * mon.politeness
        acc["rho"] += weight * mon.rho
    if total_w == 0:
        return CultureState()
    return CultureState(
        valence=acc["valence"] / total_w,
        arousal=acc["arousal"] / total_w,
        intimacy=acc["intimacy"] / total_w,
        politeness=acc["politeness"] / total_w,
        rho=acc["rho"] / total_w,
    )


def compute_spike_weight(monuments: Sequence[MemoryMonument]) -> float:
    if not monuments:
        return 0.0
    max_sal = max(mon.salience for mon in monuments)
    return min(MAX_SPIKE_WEIGHT, 0.3 + 0.6 * max_sal)


def compute_culture_state(
    context: ContextPayload,
    *,
    storage: Optional[CultureFieldStorage] = None,
    monument_storage: Optional[MonumentStorage] = None,
    monuments: Optional[Sequence[MemoryMonument]] = None,
) -> CultureState:
    ctx = _as_context(context)
    climate_store = storage or _CLIMATE_STORAGE
    c_state = _get_state(ctx.culture_tag, climate_store.culture)
    p_state = _get_state(ctx.place_id, climate_store.place)
    r_state = _get_state(ctx.partner_id, climate_store.partner)
    base = CultureState(
        valence=_avg3(c_state.valence, p_state.valence, r_state.valence),
        arousal=_avg3(c_state.arousal, p_state.arousal, r_state.arousal),
        intimacy=_avg3(c_state.intimacy, p_state.intimacy, r_state.intimacy),
        politeness=_avg3(c_state.politeness, p_state.politeness, r_state.politeness),
        rho=_avg3(c_state.rho, p_state.rho, r_state.rho),
    )
    if monuments is None:
        monuments = query_matching_monuments(context=ctx, storage=monument_storage)
    if not monuments:
        return base
    spike = aggregate_monuments(monuments)
    weight = compute_spike_weight(monuments)
    return CultureState(
        valence=(1 - weight) * base.valence + weight * spike.valence,
        arousal=(1 - weight) * base.arousal + weight * spike.arousal,
        intimacy=(1 - weight) * base.intimacy + weight * spike.intimacy,
        politeness=(1 - weight) * base.politeness + weight * spike.politeness,
        rho=(1 - weight) * base.rho + weight * spike.rho,
    )


def culture_to_behavior(state: CultureState) -> BehaviorMod:
    politeness = state.politeness
    intimacy = state.intimacy
    valence = state.valence
    arousal = state.arousal
    tone = "neutral"
    if politeness > 0.5:
        tone = "polite"
    elif intimacy > 0.6 and valence > 0:
        tone = "casual"
    empathy = max(0.2, min(0.9, 0.5 + valence * 0.3 + arousal * 0.1))
    directness = max(0.1, min(0.9, 0.6 - politeness * 0.3 + arousal * 0.2))
    joke_ratio = max(0.0, min(0.8, 0.3 + valence * 0.3 - politeness * 0.2))
    return BehaviorMod(tone=tone, empathy_level=empathy, directness=directness, joke_ratio=joke_ratio)


__all__ = [
    "BehaviorMod",
    "CultureContext",
    "CultureFieldStorage",
    "CultureState",
    "MemoryMonument",
    "MonumentKind",
    "MonumentStorage",
    "aggregate_monuments",
    "compute_culture_state",
    "compute_monument_score",
    "compute_spike_weight",
    "culture_to_behavior",
    "infer_monument_kind",
    "promote_to_monument_if_needed",
    "query_matching_monuments",
    "rescan_events_for_monuments",
    "update_climate_from_event",
    "use_climate_storage",
    "use_monument_storage",
]
