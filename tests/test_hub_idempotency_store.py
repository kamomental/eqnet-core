from __future__ import annotations

from eqnet.hub.idempotency import InMemoryIdempotencyStore, NoopIdempotencyStore


def test_noop_store_allows_duplicates() -> None:
    store = NoopIdempotencyStore()
    assert store.check_and_reserve("k")
    assert store.check_and_reserve("k")


def test_inmemory_store_blocks_duplicate_reserve() -> None:
    store = InMemoryIdempotencyStore()
    assert store.check_and_reserve("k")
    assert not store.check_and_reserve("k")
    store.mark_failed("k", "error")
    assert store.check_and_reserve("k")
