from pathlib import Path

from inner_os.evaluation import (
    build_codex_eval_pytest_command,
    default_codex_eval_environment,
    find_codex_eval_suite,
    render_codex_architecture_summary_markdown,
    render_codex_eval_plan_markdown,
    render_codex_onboarding_markdown,
    selected_codex_eval_suites,
)


def test_default_codex_eval_environment_has_onboarding_and_core_suites() -> None:
    environment = default_codex_eval_environment(Path.cwd())

    assert environment.onboarding
    assert any(entry.path == "AGENTS.md" for entry in environment.onboarding)
    assert any(suite.name == "continuity" for suite in environment.suites)
    assert any(suite.name == "deep_talk" for suite in environment.suites)
    assert environment.architecture_template_path.endswith("architecture-summary.md")


def test_build_codex_eval_pytest_command_contains_targets() -> None:
    environment = default_codex_eval_environment(Path.cwd())
    suite = find_codex_eval_suite(environment, "continuity")
    assert suite is not None

    command = build_codex_eval_pytest_command(suite, python_executable="python")

    assert command[:3] == ("python", "-m", "pytest")
    assert "tests/test_multi_turn_deep_talk_surface.py" in command
    assert command[-1] == "-q"


def test_render_codex_markdown_includes_read_order_and_suite_commands() -> None:
    environment = default_codex_eval_environment(Path.cwd())

    onboarding = render_codex_onboarding_markdown(environment)
    plan = render_codex_eval_plan_markdown(
        environment,
        suite_names=("continuity",),
        python_executable="python",
    )

    assert "AGENTS.md" in onboarding
    assert "inner_os/README.md" in onboarding
    assert "continuity" in plan
    assert "tests/test_multi_turn_deep_talk_surface.py" in plan


def test_render_architecture_summary_mentions_template_and_runtime() -> None:
    environment = default_codex_eval_environment(Path.cwd())

    summary = render_codex_architecture_summary_markdown(environment)

    assert "Architecture Summary" in summary
    assert "start_hub.py" in summary
    assert "runtime.py" in summary
    assert "template source:" in summary


def test_selected_codex_eval_suites_can_filter() -> None:
    environment = default_codex_eval_environment(Path.cwd())

    suites = selected_codex_eval_suites(environment, ("deep_talk",))

    assert len(suites) == 1
    assert suites[0].name == "deep_talk"
