from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Tuple


ROOT = Path(__file__).resolve().parent


def apply_litex_rules(
    controls: Any,
    metrics: Mapping[str, Any],
    cfg: Mapping[str, Any] | None = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Apply Litex rewrite rules to the EQNet control state.

    Parameters
    ----------
    controls:
        Control structure (dict or dataclass) produced by PolicyHead.
    metrics:
        Additional context (Sigma, love_mode, RQA, etc.) required by rules.
    cfg:
        Optional configuration mapping (mirrors `config/plugins.yaml`).

    Returns
    -------
    new_controls, meta
        `new_controls` has the same type as the input. `meta` contains
        execution details (`status`, `trace`, `reason`).
    """

    try:
        from pylitex import Rewriter, parse_rules  # type: ignore
    except ImportError:
        return controls, {"status": "skipped", "reason": "pylitex_not_installed"}

    config = cfg or {}
    litex_cfg = config.get("litex", {})
    trace_enabled = bool(litex_cfg.get("trace", False))
    max_steps = int(litex_cfg.get("max_steps", 8))

    rule_root = Path(config.get("rewrite_rules", ROOT / "rewrite_rules"))
    if rule_root.is_file():
        rule_paths = [rule_root]
    else:
        pattern = config.get("rule_glob", "*.ltx")
        rule_paths = sorted(rule_root.glob(pattern))

    if not rule_paths:
        return controls, {"status": "skipped", "reason": "no_rules"}

    try:
        rule_buffer = "\n\n".join(p.read_text(encoding="utf-8") for p in rule_paths)
        rule_set = parse_rules(rule_buffer)
        rewriter = Rewriter(rule_set, trace=trace_enabled, max_steps=max_steps)
    except Exception as exc:  # pragma: no cover - defensive
        return controls, {"status": "error", "reason": f"litex_init_failed: {exc}"}

    state_dict = _merge_state(controls, metrics)

    try:
        rewritten, trace = rewriter.rewrite(state_dict)
    except AttributeError:
        # Older pylitex versions may expose `normalize`.
        try:
            rewritten, trace = rewriter.normalize(state_dict, return_trace=True)
        except Exception as exc:  # pragma: no cover - defensive
            return controls, {"status": "error", "reason": f"rewriter_missing_api: {exc}"}
    except Exception as exc:  # pragma: no cover - defensive
        return controls, {"status": "error", "reason": f"rewrite_failed: {exc}"}

    new_controls = _reconstruct(controls, rewritten)
    meta = {
        "status": "applied",
        "rules": [str(p) for p in rule_paths],
        "trace": trace if trace_enabled else None,
    }
    return new_controls, meta


# ---------------------------------------------------------------------------
# Helpers

def _merge_state(controls: Any, metrics: Mapping[str, Any]) -> Dict[str, Any]:
    control_dict = _to_dict(controls)
    merged = dict(metrics)
    merged.update(control_dict)
    return merged


def _to_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return dict(obj)
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if hasattr(obj, "__dict__"):
        return {
            key: value
            for key, value in vars(obj).items()
            if not key.startswith("_")
        }
    raise TypeError(f"Unsupported controls type: {type(obj)!r}")


def _reconstruct(original: Any, data: Mapping[str, Any]) -> Any:
    if isinstance(original, dict):
        return {**original, **{k: data.get(k, v) for k, v in original.items()}}
    if dataclasses.is_dataclass(original):
        values = {
            field.name: data.get(field.name, getattr(original, field.name))
            for field in dataclasses.fields(original)
        }
        return type(original)(**values)
    if hasattr(original, "__dict__"):
        for key, value in data.items():
            if hasattr(original, key):
                setattr(original, key, value)
        return original
    return original
