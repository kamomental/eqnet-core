from __future__ import annotations


def test_overlay_latest_headers(client):
    res = client.get("/overlay/latest")
    assert res.status_code == 200
    assert res.headers.get("cache-control") == "no-store"
    assert res.headers.get("pragma") == "no-cache"


def test_overlay_fragment_headers(client):
    res = client.get("/partials/overlay/latest", params={"_ts": 123})
    assert res.status_code == 200
    assert res.headers.get("cache-control") == "no-store"
    assert res.headers.get("pragma") == "no-cache"
