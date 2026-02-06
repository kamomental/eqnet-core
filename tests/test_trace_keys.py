from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace

from eqnet.hub.trace_keys import (
    get_or_create_episode_id,
    resolve_day_key_from_as_of,
    resolve_day_key_from_date,
    resolve_day_key_from_moment,
)


def test_day_key_resolvers_are_utc_and_normalized() -> None:
    stamp = datetime(2025, 12, 14, 23, 59, 59, tzinfo=timezone.utc)
    assert resolve_day_key_from_moment(stamp) == "2025-12-14"
    assert resolve_day_key_from_date(date(2025, 12, 14)) == "2025-12-14"
    assert resolve_day_key_from_as_of("20251214") == "2025-12-14"
    assert resolve_day_key_from_as_of("2025-12-14") == "2025-12-14"


def test_episode_id_provider_is_present_and_stable_for_same_seed() -> None:
    ts = datetime(2025, 12, 14, 10, 0, tzinfo=timezone.utc)
    entry = SimpleNamespace(
        session_id="session-x",
        scenario_id="scenario-y",
        turn_id=9,
        timestamp=ts,
    )
    first = get_or_create_episode_id(entry)
    second = get_or_create_episode_id(entry)
    assert first.startswith("ep-")
    assert first == second

