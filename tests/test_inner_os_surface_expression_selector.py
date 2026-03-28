# -*- coding: utf-8 -*-

from emot_terrain_lab.hub.runtime import EmotionalHubRuntime, RuntimeConfig
from inner_os.expression.surface_expression_selector import (
    build_surface_expression_candidates,
    choose_surface_expression,
)


def test_surface_expression_selector_prefers_casual_thread_reopen_return_for_casual_register() -> None:
    selected = choose_surface_expression(
        build_surface_expression_candidates(
            [
                "続きは、また話せそうなときに。",
                "その先は、また話せそうなときに。",
                "残りは、また話せそうなときで大丈夫です。",
            ],
            candidate_profile="thread_reopen_return",
        ),
        cultural_register="casual_shared",
        group_register="one_to_one",
        sentence_temperature="warm",
        recent_history=(),
    )

    assert selected == "続きは、また話せそうなときに。"


def test_surface_expression_selector_prefers_polite_thread_reopen_return_for_formal_register() -> None:
    selected = choose_surface_expression(
        build_surface_expression_candidates(
            [
                "続きは、また話せそうなときに。",
                "その先は、また話せそうなときに。",
                "残りは、また話せそうなときで大丈夫です。",
            ],
            candidate_profile="thread_reopen_return",
        ),
        cultural_register="careful_polite",
        group_register="one_to_one",
        sentence_temperature="gentle",
        recent_history=(),
    )

    assert selected == "残りは、また話せそうなときで大丈夫です。"


def test_runtime_thread_reopen_compaction_uses_register_conditioned_return_phrase() -> None:
    runtime = EmotionalHubRuntime(RuntimeConfig(use_eqnet_core=True))
    runtime._response_locale = lambda: "ja-JP"  # type: ignore[attr-defined]

    compact = runtime._compact_inner_os_sequence_text(
        [
            {
                "act": "reopen_from_anchor",
                "text": "「あの約束」のところなら、いま話せる分だけで大丈夫です。",
            },
            {
                "act": "leave_return_point",
                "text": "続きは、また話せそうなときに。",
            },
        ],
        surface_profile={
            "response_length": "short",
            "cultural_register": "careful_polite",
            "group_register": "one_to_one",
            "sentence_temperature": "gentle",
        },
    )

    assert compact == "あの約束のところなら、いま話せるぶんだけで。 残りは、また話せそうなときで大丈夫です。"


def test_surface_expression_selector_prefers_casual_continuity_opening_for_casual_register() -> None:
    selected = choose_surface_expression(
        build_surface_expression_candidates(
            [
                "うん、話せるところからで大丈夫だよ。急がなくていいよ。",
                "話せるところからで大丈夫です。急がなくていいです。",
                "まずは話せるぶんだけで大丈夫です。",
            ],
            candidate_profile="continuity_opening",
        ),
        cultural_register="casual_shared",
        group_register="one_to_one",
        sentence_temperature="warm",
        recent_history=(),
    )

    assert selected == "うん、話せるところからで大丈夫だよ。急がなくていいよ。"


def test_surface_expression_selector_prefers_measured_continuity_opening_for_formal_register() -> None:
    selected = choose_surface_expression(
        build_surface_expression_candidates(
            [
                "うん、話せるところからで大丈夫だよ。急がなくていいよ。",
                "話せるところからで大丈夫です。急がなくていいです。",
                "まずは話せるぶんだけで大丈夫です。",
            ],
            candidate_profile="continuity_opening",
        ),
        cultural_register="careful_polite",
        group_register="one_to_one",
        sentence_temperature="gentle",
        recent_history=(),
    )

    assert selected in {
        "話せるところからで大丈夫です。急がなくていいです。",
        "まずは話せるぶんだけで大丈夫です。",
    }
