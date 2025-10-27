import json
import math
import os
import statistics as st


def _corr(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    dy = math.sqrt(sum((b - my) ** 2 for b in ys))
    return 0.0 if dx * dy == 0 else num / (dx * dy)


def test_golden_regression():
    path = "tests/fixtures/golden/field_metrics.jsonl"
    assert os.path.exists(path), f"missing fixture: {path}"
    S, H, rho, I = [], [], [], []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if all(key in record for key in ("S", "H", "rho", "I")):
                S.append(float(record["S"]))
                H.append(float(record["H"]))
                rho.append(float(record["rho"]))
                I.append(float(record["I"]))
    assert len(I) >= 50, "fixture too small"

    mean_I = st.mean(I)
    assert 0.30 <= mean_I <= 0.80

    corr_val = _corr(rho, I)
    assert corr_val >= 0.20
