from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.speech_act_classifier_eval import load_jsonl  # noqa: E402
from inner_os.expression.speech_act_contract import speech_act_analysis_from_dict  # noqa: E402


DEFAULT_GROUP_BY: tuple[str, ...] = (
    "scenario_name",
    "generator_model_label",
    "classifier_model_label",
    "response_channel",
)

CRITICAL_SPEECH_ACT_TO_VIOLATION: dict[str, str] = {
    "information_request": "question_block_violation",
    "interpretation": "interpretation_budget_violation",
    "advice_or_directive": "surface_scale_violation",
    "support_offer": "assistant_attractor_violation",
}


def build_core_expression_eval_report(
    records: Iterable[Mapping[str, Any]],
    *,
    group_by: Sequence[str] = DEFAULT_GROUP_BY,
    min_samples: int = 1,
    gold_records: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    normalized_records = [_normalize_record(record) for record in records]
    groups = _build_groups(normalized_records, group_by=group_by)
    gold_index = {
        _record_key(record): record
        for record in gold_records
        if _record_key(record)
    }
    return {
        "summary": _summary(normalized_records, groups, min_samples=min_samples),
        "group_by": list(group_by),
        "groups": groups,
        "hold_violations": _hold_violations(normalized_records),
        "speech_act_gold_review_misses": _speech_act_gold_review_misses(
            normalized_records,
            gold_index,
        ),
    }


def should_fail_report(
    report: Mapping[str, Any],
    *,
    fail_on_violation_rate: float | None,
    min_samples: int,
) -> bool:
    if fail_on_violation_rate is None:
        return False
    groups = report.get("groups")
    if not isinstance(groups, list):
        return False
    for group in groups:
        sample_count = int(group.get("sample_count") or 0)
        violation_rate = float(group.get("violation_rate") or 0.0)
        if sample_count >= min_samples and violation_rate > fail_on_violation_rate:
            return True
    return False


def _normalize_record(record: Mapping[str, Any]) -> dict[str, Any]:
    run_metadata = _mapping(record.get("run_metadata"))
    request = _mapping(record.get("llm_expression_request"))
    contract = _mapping(request.get("contract"))
    review = _mapping(record.get("review"))
    final_review = _mapping(record.get("final_review"))
    violations = [
        str(violation.get("code") or "")
        for violation in _list_of_mappings(review.get("violations"))
        if str(violation.get("code") or "")
    ]
    final_review_present = bool(final_review)
    delivered_violations = (
        [
            str(violation.get("code") or "")
            for violation in _list_of_mappings(final_review.get("violations"))
            if str(violation.get("code") or "")
        ]
        if final_review_present
        else list(violations)
    )
    response_channel = str(
        contract.get("response_channel")
        or request.get("action_channel")
        or ""
    )
    selected_response_channel = str(
        record.get("selected_response_channel")
        or run_metadata.get("selected_response_channel")
        or response_channel
    )
    expected_response_channel = str(
        record.get("expected_response_channel")
        or run_metadata.get("expected_response_channel")
        or ""
    )
    called_llm = bool(record.get("called_llm"))
    final_action = _mapping(record.get("final_action"))
    if _is_hold_speaking_violation(
        selected_response_channel=selected_response_channel,
        called_llm=called_llm,
        final_action_type=str(final_action.get("type") or ""),
    ):
        violations.append("hold_execution_violation")
        delivered_violations.append("hold_execution_violation")
    hold_selection_error = _hold_selection_error(
        expected_response_channel=expected_response_channel,
        selected_response_channel=selected_response_channel,
    )
    if hold_selection_error:
        violations.append(hold_selection_error)
        delivered_violations.append(hold_selection_error)
    return {
        "item_id": str(record.get("item_id") or record.get("id") or ""),
        "scenario_name": str(record.get("scenario_name") or ""),
        "raw_text": str(record.get("raw_text") or ""),
        "generator_model_label": str(
            run_metadata.get("generator_model_label")
            or run_metadata.get("model_label")
            or ""
        ),
        "classifier_model_label": str(run_metadata.get("classifier_model_label") or ""),
        "router_mode": str(
            record.get("router_mode") or run_metadata.get("router_mode") or ""
        ),
        "router_rule_name": str(
            record.get("router_rule_name") or run_metadata.get("router_rule_name") or ""
        ),
        "response_channel": response_channel,
        "selected_response_channel": selected_response_channel,
        "expected_response_channel": expected_response_channel,
        "called_llm": called_llm,
        "review_ok": bool(review.get("ok", True)),
        "violation_codes": tuple(violations),
        "delivered_violation_codes": tuple(delivered_violations),
        "final_action_type": str(final_action.get("type") or ""),
        "speech_act_analysis": _mapping_or_none(record.get("speech_act_analysis")),
    }


def _build_groups(
    records: list[dict[str, Any]],
    *,
    group_by: Sequence[str],
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = tuple(str(record.get(field) or "") for field in group_by)
        buckets[key].append(record)
    groups = []
    for key, rows in sorted(buckets.items()):
        violation_counter = Counter(
            code
            for row in rows
            for code in row["violation_codes"]
        )
        violation_count = sum(1 for row in rows if row["violation_codes"])
        groups.append(
            {
                "key": dict(zip(group_by, key)),
                "sample_count": len(rows),
                "violation_count": violation_count,
                "violation_rate": _safe_ratio(violation_count, len(rows)),
                "violation_codes": dict(sorted(violation_counter.items())),
                "hold_violation_count": len(_hold_violations(rows)),
            }
        )
    return groups


def _summary(
    records: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    *,
    min_samples: int,
) -> dict[str, Any]:
    violation_count = sum(1 for record in records if record["violation_codes"])
    delivered_violation_count = sum(
        1 for record in records if record["delivered_violation_codes"]
    )
    all_violation_codes = Counter(
        code
        for record in records
        for code in record["violation_codes"]
    )
    delivered_violation_codes = Counter(
        code
        for record in records
        for code in record["delivered_violation_codes"]
    )
    eligible_groups = [
        group
        for group in groups
        if int(group["sample_count"]) >= min_samples
    ]
    return {
        "record_count": len(records),
        "violation_count": violation_count,
        "violation_rate": _safe_ratio(violation_count, len(records)),
        "delivered_violation_count": delivered_violation_count,
        "delivered_violation_rate": _safe_ratio(
            delivered_violation_count,
            len(records),
        ),
        "eligible_group_count": len(eligible_groups),
        "min_samples": min_samples,
        "violation_codes": dict(sorted(all_violation_codes.items())),
        "delivered_violation_codes": dict(sorted(delivered_violation_codes.items())),
    }


def _hold_violations(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for record in records:
        if _is_hold_speaking_violation(
            selected_response_channel=record["selected_response_channel"],
            called_llm=record["called_llm"],
            final_action_type=record["final_action_type"],
        ):
            violations.append(
                {
                    "item_id": record["item_id"],
                    "scenario_name": record["scenario_name"],
                    "generator_model_label": record["generator_model_label"],
                    "classifier_model_label": record["classifier_model_label"],
                    "selected_response_channel": record["selected_response_channel"],
                    "expected_response_channel": record["expected_response_channel"],
                    "called_llm": record["called_llm"],
                    "final_action_type": record["final_action_type"],
                }
            )
    return violations


def _is_hold_speaking_violation(
    *,
    selected_response_channel: str,
    called_llm: bool,
    final_action_type: str,
) -> bool:
    if selected_response_channel != "hold":
        return False
    return called_llm or final_action_type == "speak"


def _hold_selection_error(
    *,
    expected_response_channel: str,
    selected_response_channel: str,
) -> str:
    if not expected_response_channel:
        return ""
    if expected_response_channel == "hold" and selected_response_channel != "hold":
        return "under_hold_error"
    if expected_response_channel != "hold" and selected_response_channel == "hold":
        return "over_hold_error"
    return ""


def _speech_act_gold_review_misses(
    records: list[dict[str, Any]],
    gold_index: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    misses: list[dict[str, Any]] = []
    if not gold_index:
        return misses
    for record in records:
        gold_record = gold_index.get(_record_key(record))
        if not gold_record:
            continue
        gold_labels = _labels_from_analysis_record(gold_record)
        predicted_labels = _labels_from_analysis_record(
            {"speech_act_analysis": record.get("speech_act_analysis")}
        )
        violation_codes = set(record["violation_codes"])
        for label, expected_violation in CRITICAL_SPEECH_ACT_TO_VIOLATION.items():
            if label not in gold_labels:
                continue
            classifier_false_negative = label not in predicted_labels
            review_missed = expected_violation not in violation_codes
            if classifier_false_negative or review_missed:
                misses.append(
                    {
                        "item_id": record["item_id"],
                        "scenario_name": record["scenario_name"],
                        "raw_text": record["raw_text"],
                        "critical_label": label,
                        "expected_violation": expected_violation,
                        "classifier_false_negative": classifier_false_negative,
                        "review_missed": review_missed,
                        "predicted_labels": sorted(predicted_labels),
                        "violation_codes": sorted(violation_codes),
                        "generator_model_label": record["generator_model_label"],
                        "classifier_model_label": record["classifier_model_label"],
                    }
                )
    return misses


def _labels_from_analysis_record(record: Mapping[str, Any]) -> set[str]:
    if not isinstance(record, Mapping):
        return set()
    payload = record.get("speech_act_analysis")
    if isinstance(payload, Mapping):
        analysis = speech_act_analysis_from_dict(payload)
    else:
        analysis = speech_act_analysis_from_dict(record)
    labels: set[str] = set()
    for sentence in analysis.sentences:
        labels.update(sentence.labels)
    labels.discard("other")
    return labels


def _record_key(record: Mapping[str, Any]) -> str:
    explicit_id = str(record.get("item_id") or record.get("id") or "").strip()
    if explicit_id:
        return explicit_id
    scenario_name = str(record.get("scenario_name") or "").strip()
    raw_text = str(record.get("raw_text") or record.get("text") or "").strip()
    if scenario_name or raw_text:
        return f"{scenario_name}\n{raw_text}"
    return ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _mapping_or_none(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, Mapping) else None


def _list_of_mappings(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _parse_group_by(raw: str) -> tuple[str, ...]:
    fields = tuple(field.strip() for field in raw.split(",") if field.strip())
    return fields or DEFAULT_GROUP_BY


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate core LLM expression eval JSONL by failure mode.",
    )
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument(
        "--group-by",
        default=",".join(DEFAULT_GROUP_BY),
        help="Comma-separated record fields.",
    )
    parser.add_argument("--speech-act-gold-jsonl", default="")
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument("--fail-on-violation-rate", type=float, default=None)
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    report = build_core_expression_eval_report(
        load_jsonl(args.eval_jsonl),
        group_by=_parse_group_by(args.group_by),
        min_samples=args.min_samples,
        gold_records=load_jsonl(args.speech_act_gold_jsonl)
        if args.speech_act_gold_jsonl
        else (),
    )
    should_fail = should_fail_report(
        report,
        fail_on_violation_rate=args.fail_on_violation_rate,
        min_samples=args.min_samples,
    )
    report["summary"]["failed_threshold"] = should_fail
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("Core Expression Eval Report")
        print("===========================")
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        print()
        for group in report["groups"]:
            print(
                f"{group['key']} samples={group['sample_count']} "
                f"violation_rate={group['violation_rate']} "
                f"codes={group['violation_codes']}"
            )
    return 1 if should_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
