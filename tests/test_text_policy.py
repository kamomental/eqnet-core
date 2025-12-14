import os

from eqnet.hub.text_policy import apply_text_policy


def test_text_policy_default_redact():
    sanitized, obs = apply_text_policy("hello", policy="redact", allow_raw_env=False)
    assert sanitized == "<redacted>"
    assert obs["len_chars"] == 5


def test_text_policy_raw_requires_env():
    sanitized, _ = apply_text_policy("hello", policy="raw", allow_raw_env=False)
    assert sanitized == "<redacted>"


def test_text_policy_raw_forced_redact_in_ci(monkeypatch):
    monkeypatch.setenv("CI", "true")
    sanitized, _ = apply_text_policy("hello", policy="raw", allow_raw_env=True)
    assert sanitized == "<redacted>"
    monkeypatch.delenv("CI")


def test_text_policy_hash_mode(monkeypatch):
    sanitized, obs = apply_text_policy("hello", policy="hash", allow_raw_env=False)
    assert sanitized == "<redacted>"
    assert "sha256" in obs
    assert obs["len_chars"] == 5


def test_text_policy_truncate_masks():
    sanitized, obs = apply_text_policy("user@test.com 555", policy="truncate", allow_raw_env=False, truncate_chars=6)
    assert sanitized.startswith("<email")
    assert "sha256" in obs
