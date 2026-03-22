from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path


def _load_script_module():
    script_path = Path("scripts") / "warm_start_model_swap.py"
    spec = importlib.util.spec_from_file_location("warm_start_model_swap", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_warm_start_model_swap_script_writes_summary_and_bundle(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()
    calls: dict[str, object] = {}

    class _FakeRuntime:
        def __init__(self, config) -> None:
            calls["config"] = config
            self._transfer_package_path = None

        def warm_start_from_transfer_package(
            self,
            package_or_path,
            *,
            target_model: str = "",
            target_base_url: str = "",
            persist_normalized: bool = False,
            prefer_target_model: bool = True,
        ):
            calls["warm_start"] = {
                "package_or_path": str(package_or_path),
                "target_model": target_model,
                "target_base_url": target_base_url,
                "persist_normalized": persist_normalized,
                "prefer_target_model": prefer_target_model,
            }
            return {
                "schema": "inner_os_transfer_package/v1",
                "target_model": target_model,
                "semantic_seed_anchor": "harbor slope",
                "initiative_followup_state": "offer_next_step",
            }

        def build_inner_os_model_swap_bundle(
            self,
            *,
            target_model: str,
            target_base_url: str = "",
            result=None,
            nightly_summary=None,
        ):
            calls["bundle"] = {
                "target_model": target_model,
                "target_base_url": target_base_url,
            }
            return {
                "schema": "inner_os_model_swap_bundle/v1",
                "target_model": target_model,
                "target_base_url": target_base_url,
                "transfer_package": {"schema": "inner_os_transfer_package/v1"},
            }

    monkeypatch.setattr(module, "EmotionalHubRuntime", _FakeRuntime)
    monkeypatch.setattr(module, "RuntimeConfig", lambda **kwargs: kwargs)

    transfer_package_path = tmp_path / "transfer_package.json"
    transfer_package_path.write_text("{}", encoding="utf-8")
    summary_path = tmp_path / "summary.json"
    bundle_path = tmp_path / "bundle.json"
    stdout = io.StringIO()

    with contextlib.redirect_stdout(stdout):
        code = module.main(
            [
                "--transfer-package",
                str(transfer_package_path),
                "--target-model",
                "qwen-3.5-coder",
                "--target-base-url",
                "http://127.0.0.1:1234/v1",
                "--summary-out",
                str(summary_path),
                "--bundle-out",
                str(bundle_path),
                "--persist-normalized",
            ]
        )

    assert code == 0
    assert calls["warm_start"]["target_model"] == "qwen-3.5-coder"
    assert calls["warm_start"]["persist_normalized"] is True
    assert calls["bundle"]["target_base_url"] == "http://127.0.0.1:1234/v1"

    written_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    written_bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    stdout_summary = json.loads(stdout.getvalue())

    assert written_summary["semantic_seed_anchor"] == "harbor slope"
    assert written_bundle["schema"] == "inner_os_model_swap_bundle/v1"
    assert stdout_summary["target_model"] == "qwen-3.5-coder"
