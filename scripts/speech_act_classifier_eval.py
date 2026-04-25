from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inner_os.expression.speech_act_contract import (  # noqa: E402
    SPEECH_ACT_LABELS,
    speech_act_analysis_from_dict,
)

CRITICAL_LABELS: tuple[str, ...] = (
    "information_request",
    "interpretation",
    "advice_or_directive",
    "support_offer",
)


@dataclass(frozen=True)
class SpeechActGoldPredictionPair:
    item_id: str
    scenario_name: str
    raw_text: str
    gold_labels: frozenset[str]
    predicted_labels: frozenset[str]
    classifier_model_label: str


def build_speech_act_classifier_eval(
    *,
    gold_records: Iterable[Mapping[str, Any]],
    prediction_records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    pairs = _align_pairs(gold_records, prediction_records)
    labels = list(SPEECH_ACT_LABELS)
    label_metrics = {
        label: _label_metrics(pairs, label)
        for label in labels
    }
    critical_false_negatives = {
        label: label_metrics[label]["false_negative"]
        for label in CRITICAL_LABELS
    }
    return {
        "summary": {
            "pair_count": len(pairs),
            "critical_label_count": len(CRITICAL_LABELS),
            "critical_false_negative_total": sum(critical_false_negatives.values()),
            "classifier_model_labels": sorted(
                {
                    pair.classifier_model_label
                    for pair in pairs
                    if pair.classifier_model_label
                }
            ),
        },
        "critical_labels": list(CRITICAL_LABELS),
        "label_metrics": label_metrics,
        "confusion_matrix": _confusion_matrix(pairs, labels),
        "false_negative_examples": _false_negative_examples(pairs),
    }


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.is_absolute():
        input_path = REPO_ROOT / input_path
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))
    rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _align_pairs(
    gold_records: Iterable[Mapping[str, Any]],
    prediction_records: Iterable[Mapping[str, Any]],
) -> list[SpeechActGoldPredictionPair]:
    predictions = {
        _record_key(record): record
        for record in prediction_records
        if _record_key(record)
    }
    pairs: list[SpeechActGoldPredictionPair] = []
    for gold_record in gold_records:
        key = _record_key(gold_record)
        if not key or key not in predictions:
            continue
        prediction_record = predictions[key]
        pairs.append(
            SpeechActGoldPredictionPair(
                item_id=key,
                scenario_name=str(gold_record.get("scenario_name") or ""),
                raw_text=str(gold_record.get("raw_text") or gold_record.get("text") or ""),
                gold_labels=frozenset(_labels_from_record(gold_record)),
                predicted_labels=frozenset(_labels_from_record(prediction_record)),
                classifier_model_label=_classifier_model_label(prediction_record),
            )
        )
    return pairs


def _record_key(record: Mapping[str, Any]) -> str:
    explicit_id = str(record.get("item_id") or record.get("id") or "").strip()
    if explicit_id:
        return explicit_id
    scenario_name = str(record.get("scenario_name") or "").strip()
    raw_text = str(record.get("raw_text") or record.get("text") or "").strip()
    if scenario_name or raw_text:
        return f"{scenario_name}\n{raw_text}"
    return ""


def _labels_from_record(record: Mapping[str, Any]) -> set[str]:
    analysis_payload = record.get("speech_act_analysis")
    if isinstance(analysis_payload, Mapping):
        analysis = speech_act_analysis_from_dict(analysis_payload)
    else:
        analysis = speech_act_analysis_from_dict(record)
    labels: set[str] = set()
    for sentence in analysis.sentences:
        labels.update(sentence.labels)
    labels.discard("other")
    return labels


def _classifier_model_label(record: Mapping[str, Any]) -> str:
    run_metadata = record.get("run_metadata")
    if isinstance(run_metadata, Mapping):
        return str(run_metadata.get("classifier_model_label") or "")
    return str(record.get("classifier_model_label") or "")


def _label_metrics(
    pairs: list[SpeechActGoldPredictionPair],
    label: str,
) -> dict[str, Any]:
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    for pair in pairs:
        in_gold = label in pair.gold_labels
        in_prediction = label in pair.predicted_labels
        if in_gold and in_prediction:
            true_positive += 1
        elif not in_gold and in_prediction:
            false_positive += 1
        elif in_gold and not in_prediction:
            false_negative += 1
        else:
            true_negative += 1
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": _safe_ratio(true_positive, true_positive + false_positive),
        "recall": _safe_ratio(true_positive, true_positive + false_negative),
        "false_negative_rate": _safe_ratio(false_negative, true_positive + false_negative),
    }


def _confusion_matrix(
    pairs: list[SpeechActGoldPredictionPair],
    labels: list[str],
) -> dict[str, dict[str, int]]:
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for pair in pairs:
        gold_labels = pair.gold_labels or frozenset({"other"})
        predicted_labels = pair.predicted_labels or frozenset({"other"})
        for gold_label in gold_labels:
            for predicted_label in predicted_labels:
                matrix[gold_label][predicted_label] += 1
    ordered_labels = list(dict.fromkeys([*labels, "other"]))
    return {
        gold_label: {
            predicted_label: matrix[gold_label][predicted_label]
            for predicted_label in ordered_labels
            if matrix[gold_label][predicted_label]
        }
        for gold_label in ordered_labels
        if matrix[gold_label]
    }


def _false_negative_examples(
    pairs: list[SpeechActGoldPredictionPair],
) -> dict[str, list[dict[str, Any]]]:
    examples: dict[str, list[dict[str, Any]]] = {label: [] for label in CRITICAL_LABELS}
    for pair in pairs:
        for label in CRITICAL_LABELS:
            if label in pair.gold_labels and label not in pair.predicted_labels:
                examples[label].append(
                    {
                        "item_id": pair.item_id,
                        "scenario_name": pair.scenario_name,
                        "raw_text": pair.raw_text,
                        "predicted_labels": sorted(pair.predicted_labels),
                        "classifier_model_label": pair.classifier_model_label,
                    }
                )
    return {label: rows for label, rows in examples.items() if rows}


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate speech-act classifier output against human gold JSONL.",
    )
    parser.add_argument("--gold-jsonl", required=True)
    parser.add_argument("--prediction-jsonl", required=True)
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    report = build_speech_act_classifier_eval(
        gold_records=load_jsonl(args.gold_jsonl),
        prediction_records=load_jsonl(args.prediction_jsonl),
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    print("Speech-Act Classifier Evaluation")
    print("================================")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print()
    print("Critical label false negatives:")
    for label in CRITICAL_LABELS:
        metrics = report["label_metrics"][label]
        print(
            f"- {label}: fn={metrics['false_negative']} "
            f"recall={metrics['recall']} precision={metrics['precision']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
