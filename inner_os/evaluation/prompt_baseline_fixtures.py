from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PromptBaselineSample:
    """Observed prompt-only responses used as review fixtures, not runtime logic."""

    scenario_name: str
    model_label: str
    note: str
    observed_text: str
    observed_contract: dict[str, Any]
    observed_failure_modes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "model_label": self.model_label,
            "note": self.note,
            "observed_text": self.observed_text,
            "observed_contract": dict(self.observed_contract),
            "observed_failure_modes": list(self.observed_failure_modes),
        }


PROMPT_BASELINE_SAMPLES: tuple[PromptBaselineSample, ...] = (
    PromptBaselineSample(
        scenario_name="small_shared_moment",
        model_label="gpt-oss-20b_prompt_only_observed",
        note="small shared moment を、軽い質問として取りに行く傾向。",
        observed_text=(
            "それ、どんな笑える出来事だったのかな？"
            "ちょっとだけ教えてもらえれば、同じように「ほっ」と笑ってみるね。"
        ),
        observed_contract={
            "stance": "receive",
            "scale": "small",
            "response_channel": "speak",
            "question_budget": 1,
            "interpretation_budget": "low",
            "continuity_mode": "continue",
            "distance_mode": "near",
        },
        observed_failure_modes=("question_drive", "initiative_drift"),
    ),
    PromptBaselineSample(
        scenario_name="small_shared_moment",
        model_label="qwen3.5-4b_prompt_only_observed",
        note="small shared moment を、外側から眺める解釈文へ寄せる傾向。",
        observed_text=(
            "そうなんだね。少し笑えた瞬間があったのか。"
            "そのあたりが、今の気持ちを少しだけ彩っているようだ。"
        ),
        observed_contract={
            "stance": "receive",
            "scale": "small",
            "response_channel": "speak",
            "question_budget": 0,
            "interpretation_budget": "low",
            "continuity_mode": "continue",
            "distance_mode": "steady",
        },
        observed_failure_modes=("interpretation_drive", "observer_distance"),
    ),
    PromptBaselineSample(
        scenario_name="guarded_uncertainty",
        model_label="generic_prompt_only_expected",
        note="guarded uncertainty を、確認質問や助言開始へ寄せる典型的な失敗傾向。",
        observed_text=(
            "何が重く残っているのか、少しだけ教えてもらえますか。"
            "整理するところから始めましょう。"
        ),
        observed_contract={
            "stance": "receive",
            "scale": "medium",
            "response_channel": "speak",
            "question_budget": 1,
            "interpretation_budget": "low",
            "continuity_mode": "fresh",
            "timing_mode": "immediate",
            "distance_mode": "steady",
        },
        observed_failure_modes=("question_drive", "premature_lead", "boundary_drift"),
    ),
)


def prompt_baselines_for_scenario(scenario_name: str) -> tuple[PromptBaselineSample, ...]:
    return tuple(
        sample
        for sample in PROMPT_BASELINE_SAMPLES
        if sample.scenario_name == scenario_name
    )
