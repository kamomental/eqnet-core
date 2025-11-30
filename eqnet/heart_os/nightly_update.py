"""Nightly promotion for Heart OS."""

from __future__ import annotations

from datetime import date, datetime
from typing import Iterable, List, Sequence

from eqnet.memory import Episode, Monument
from eqnet.memory.episodes import extract_ids, most_common
from eqnet.memory.impact import compute_impact
from eqnet.memory.moment_knn import MomentKNNIndex

MONUMENT_TH = 0.3
EPISODE_TH = 0.45


def get_today_moments(moment_log: Sequence[object], day: date) -> List[object]:
    moments: List[object] = []
    for moment in moment_log:
        stamp = _moment_date(moment)
        if stamp == day:
            moments.append(moment)
    return moments


def _moment_date(moment: object) -> date | None:
    stamp = getattr(moment, "timestamp", None)
    if stamp is not None:
        if isinstance(stamp, datetime):
            return stamp.date()
        if isinstance(stamp, str):
            try:
                return datetime.fromisoformat(stamp).date()
            except ValueError:
                pass
    ts = getattr(moment, "ts", None)
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(float(ts)).date()
    except (TypeError, ValueError, OSError):
        return None


def _moment_sort_key(moment: object) -> float:
    stamp = getattr(moment, "timestamp", None)
    if isinstance(stamp, datetime):
        return stamp.timestamp()
    if isinstance(stamp, str):
        try:
            return datetime.fromisoformat(stamp).timestamp()
        except ValueError:
            pass
    ts = getattr(moment, "ts", None)
    try:
        return float(ts)
    except (TypeError, ValueError):
        return 0.0


def build_episode(segment: Sequence[object], day: date, impacts: Sequence[float]) -> Episode:
    topics = [getattr(m, "topic", "") for m in segment]
    emotions = [getattr(m, "emotion_tag", "") for m in segment]
    place_ids = [getattr(m, "place_id", "") for m in segment if getattr(m, "place_id", None)]

    partners: List[str] = []
    for moment in segment:
        ids = getattr(moment, "partner_ids", None) or []
        partners.extend(ids)

    summary = _summarize_segment(segment)
    return Episode(
        id=_gen_id(),
        date=day,
        moments=[getattr(m, "id", "") for m in segment],
        place_ids=extract_ids(place_ids),
        partner_ids=extract_ids(partners),
        dominant_topic=most_common(topics),
        dominant_emotion=most_common(emotions, default="neutral"),
        impact=max(impacts),
        summary=summary,
    )


def build_monument(moment: object, impact: float) -> Monument:
    return Monument(
        id=_gen_id(),
        place_id=str(getattr(moment, "place_id", "unknown")),
        partner_ids=list(getattr(moment, "partner_ids", []) or []),
        culture_tag=str(getattr(moment, "culture_tag", getattr(moment, "topic", "misc"))),
        core_emotion=str(getattr(moment, "emotion_tag", "neutral")),
        importance=impact,
        summary=f"{getattr(moment, 'place_id', 'somewhere')} での出来事: {getattr(moment, 'user_text', '')}",
        episodes=[],
    )


def nightly_promote(
    *,
    moment_log: Sequence[object],
    day: date,
    terrain_state,
    store,
    knn_index: MomentKNNIndex | None = None,
) -> None:
    today = get_today_moments(moment_log, day)
    today = sorted(today, key=_moment_sort_key)

    impacts = [compute_impact(m, terrain_state) for m in today]

    for moment, impact in zip(today, impacts):
        if impact >= MONUMENT_TH and terrain_state.should_create_monument(moment):
            mon = build_monument(moment, impact)
            store.save_monument(mon)
            terrain_state.register_monument(mon)

    segment: List[object] = []
    segment_impacts: List[float] = []
    for moment, impact in zip(today, impacts):
        if impact >= EPISODE_TH:
            segment.append(moment)
            segment_impacts.append(impact)
            continue
        if segment:
            _finalize_episode(segment, segment_impacts, day, terrain_state, store)
            segment = []
            segment_impacts = []
    if segment:
        _finalize_episode(segment, segment_impacts, day, terrain_state, store)

    if knn_index is not None:
        knn_index.decay_all(0.9)


def _finalize_episode(segment: List[object], impacts: List[float], day: date, terrain_state, store) -> None:
    episode = build_episode(segment, day, impacts)
    store.save_episode(episode)
    terrain_state.register_episode(episode)


def _summarize_segment(segment: Sequence[object]) -> str:
    if not segment:
        return ""
    start = getattr(segment[0], "user_text", "")
    end = getattr(segment[-1], "user_text", "")
    topic = most_common([getattr(m, "topic", "") for m in segment])
    return f"{topic} を巡る一連の出来事。『{start}』ではじまり、『{end}』で終わった。"


def _gen_id() -> str:
    import uuid

    return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from datetime import date

    from eqnet.memory import MemoryStore, TerrainState
    from eqnet.memory.moment_knn import MomentKNNIndex
    from eqnet.logs.moment_log import iter_moment_entries

    parser = argparse.ArgumentParser(description="Run nightly promotion (L1 → Episodes/Monuments)")
    parser.add_argument("--log", required=True, help="Path to today's MomentLog JSONL file")
    parser.add_argument(
        "--day",
        default=None,
        help="Target date YYYY-MM-DD (defaults to today)",
    )
    parser.add_argument(
        "--memory-dir",
        default="eqnet_data/memory",
        help="Directory containing episodes.jsonl / monuments.jsonl",
    )
    args = parser.parse_args()

    target_day = date.fromisoformat(args.day) if args.day else date.today()
    log_path = Path(args.log)
    if not log_path.exists():
        raise SystemExit(f"moment log not found: {log_path}")

    memory_dir = Path(args.memory_dir)
    store = MemoryStore(memory_dir)
    episodes, monuments = store.load_all()
    terrain_state = TerrainState(episodes=episodes, monuments=monuments)

    moment_log = list(iter_moment_entries(log_path))
    knn_index = MomentKNNIndex()

    print(f"[Nightly] Running for {target_day} with {len(moment_log)} moments from {log_path}")
    nightly_promote(
        moment_log=moment_log,
        day=target_day,
        terrain_state=terrain_state,
        store=store,
        knn_index=knn_index,
    )
    print("[Nightly] Completed")
