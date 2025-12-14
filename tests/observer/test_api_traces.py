from __future__ import annotations


def test_api_list_traces_contract(client):
    res = client.get("/api/traces/2025-12-13")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert data, "expected at least one trace file"
    item = data[0]
    for key in ("file", "source_loop", "pid", "size_bytes", "turn_range"):
        assert key in item
    assert set(item["turn_range"].keys()) == {"min", "max"}


def test_api_trace_page_contract_and_redaction(client):
    res = client.get("/api/traces/2025-12-13/runtime-abc-1234.jsonl", params={"offset": 0, "limit": 2})
    assert res.status_code == 200
    payload = res.json()
    assert set(payload.keys()) == {"data", "next_offset"}
    rows = payload["data"]
    assert len(rows) == 2

    row = rows[0]
    assert {"line_no", "timestamp_ms", "turn_id", "redacted"} <= set(row.keys())
    redacted = row["redacted"]
    assert "raw_transcript" not in redacted
    assert "raw_transcript_char_len" in redacted
