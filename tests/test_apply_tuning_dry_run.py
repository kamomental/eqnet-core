import json
import subprocess
import sys

import yaml


def test_apply_tuning_dry_run(tmp_path):
    cfg_path = tmp_path / "runtime.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"ignition": {"theta_on": 0.62, "theta_off": 0.48}}, sort_keys=False),
        encoding="utf-8",
    )
    nightly_path = tmp_path / "nightly.json"
    nightly_path.write_text(
        json.dumps({"tuning_suggestion": {"theta_on": 0.6, "theta_off": 0.46}}, ensure_ascii=False),
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        "scripts/apply_nightly_tuning.py",
        "--nightly",
        str(nightly_path),
        "--config",
        str(cfg_path),
    ]
    subprocess.run(cmd, check=True)

    cfg_after = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert cfg_after["ignition"]["theta_on"] == 0.62
    assert cfg_after["ignition"]["theta_off"] == 0.48
