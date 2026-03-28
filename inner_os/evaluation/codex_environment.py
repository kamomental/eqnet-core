from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class CodexOnboardingEntry:
    label: str
    path: str
    purpose: str


@dataclass(frozen=True)
class CodexEvalSuite:
    name: str
    score_key: str
    goal: str
    test_targets: tuple[str, ...]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class CodexEvalEnvironment:
    repo_root: str
    report_path: str
    architecture_template_path: str
    onboarding: tuple[CodexOnboardingEntry, ...] = field(default_factory=tuple)
    suites: tuple[CodexEvalSuite, ...] = field(default_factory=tuple)


def default_codex_eval_environment(repo_root: str | Path) -> CodexEvalEnvironment:
    root = Path(repo_root)
    return CodexEvalEnvironment(
        repo_root=str(root),
        report_path=str(root / "reports" / "codex_eval" / "latest.md"),
        architecture_template_path=str(
            root
            / "skills"
            / "codebase-understanding"
            / "templates"
            / "architecture-summary.md"
        ),
        onboarding=(
            CodexOnboardingEntry(
                label="作業規約",
                path="AGENTS.md",
                purpose="制約、設計原則、日本語運用、完了条件を最初に読む。",
            ),
            CodexOnboardingEntry(
                label="Inner OS 概観",
                path="inner_os/README.md",
                purpose="何が reusable core で、どこまでが同一存在の継続機構かを掴む。",
            ),
            CodexOnboardingEntry(
                label="Core docs map",
                path="docs/core/README.md",
                purpose="主要 docs と現在の評価・issue map への入口として使う。",
            ),
            CodexOnboardingEntry(
                label="評価基準",
                path="docs/core/evaluation_criteria.md",
                purpose="内部評価軸と外部評価軸を確認する。",
            ),
            CodexOnboardingEntry(
                label="評価目標",
                path="docs/core/evaluation_targets.md",
                purpose="現在地、目標値、改善ループの優先順を確認する。",
            ),
            CodexOnboardingEntry(
                label="機序 issue map",
                path="docs/core/mechanism_issue_map.md",
                purpose="どの失敗をどの機序で直すかを見る。",
            ),
        ),
        suites=(
            CodexEvalSuite(
                name="general_surface",
                score_key="一般会話自然さ",
                goal="一般層に通じる短い日本語と generic fallback 抑制を確認する。",
                test_targets=(
                    "tests/test_runtime_route_prompts.py",
                    "tests/test_runtime_short_content_sequence.py",
                    "tests/test_runtime_deep_reflection_compaction.py",
                    "tests/test_human_output_examples.py",
                ),
                notes=(
                    "相談窓口文体へ寄りすぎないことを見る。",
                    "deep reflection の短い止め方の自然さも含む。",
                ),
            ),
            CodexEvalSuite(
                name="deep_talk",
                score_key="深い話の受け止め",
                goal="内容反映が先に立ち、問いすぎずに止まれるかを見る。",
                test_targets=(
                    "tests/test_inner_os_deep_disclosure.py",
                    "tests/test_runtime_deep_reflection_compaction.py",
                    "tests/test_runtime_short_content_sequence.py",
                ),
                notes=(
                    "green_reflection_hold と contact reflection の切替を見る。",
                ),
            ),
            CodexEvalSuite(
                name="continuity",
                score_key="数ターン継続",
                goal="論点核の保持、anchor reopening、generic continuation 抑制を確認する。",
                test_targets=(
                    "tests/test_multi_turn_deep_talk_surface.py",
                    "tests/test_inner_os_anchor_normalization.py",
                    "tests/test_inner_os_discussion_thread_registry.py",
                    "tests/test_inner_os_interaction_constraints.py",
                    "tests/test_runtime_route_prompts.py",
                ),
                notes=(
                    "表面文ではなく thread anchor を保持できるかを見る。",
                ),
            ),
            CodexEvalSuite(
                name="mechanisms",
                score_key="機序の実在性",
                goal="Green / contact / residual / boundary が typed に通っているかを見る。",
                test_targets=(
                    "tests/test_inner_os_green_kernel_contracts.py",
                    "tests/test_runtime_process_turn_hooks.py",
                    "tests/test_inner_os_conversational_architecture.py",
                ),
                notes=(
                    "表出だけでなく hooks と shared field まで確認する。",
                ),
            ),
        ),
    )


def find_codex_eval_suite(
    environment: CodexEvalEnvironment,
    name: str,
) -> CodexEvalSuite | None:
    target = str(name or "").strip().lower()
    if not target:
        return None
    for suite in environment.suites:
        if suite.name.lower() == target:
            return suite
    return None


def build_codex_eval_pytest_command(
    suite: CodexEvalSuite,
    *,
    python_executable: str = "python",
) -> tuple[str, ...]:
    return (
        python_executable,
        "-m",
        "pytest",
        *suite.test_targets,
        "-q",
    )


def render_codex_onboarding_markdown(environment: CodexEvalEnvironment) -> str:
    lines = [
        "# Codex Onboarding Environment",
        "",
        "## 読み順",
        "",
    ]
    for index, entry in enumerate(environment.onboarding, start=1):
        lines.extend(
            (
                f"{index}. `{entry.path}`",
                f"   - {entry.label}",
                f"   - {entry.purpose}",
            )
        )
    lines.extend(
        (
            "",
            "## 評価スイート",
            "",
        )
    )
    for suite in environment.suites:
        lines.extend(
            (
                f"- `{suite.name}`",
                f"  - score key: {suite.score_key}",
                f"  - goal: {suite.goal}",
                f"  - tests: {', '.join(suite.test_targets)}",
            )
        )
        for note in suite.notes:
            lines.append(f"  - note: {note}")
    return "\n".join(lines).strip() + "\n"


def render_codex_eval_plan_markdown(
    environment: CodexEvalEnvironment,
    *,
    suite_names: Sequence[str] | None = None,
    python_executable: str = "python",
) -> str:
    names = [str(item).strip().lower() for item in suite_names or () if str(item).strip()]
    suites = (
        [suite for suite in environment.suites if suite.name.lower() in set(names)]
        if names
        else list(environment.suites)
    )
    lines = [
        "# Codex Evaluation Plan",
        "",
        f"- repo root: `{environment.repo_root}`",
        f"- report path: `{environment.report_path}`",
        "",
    ]
    for suite in suites:
        command = " ".join(
            build_codex_eval_pytest_command(
                suite,
                python_executable=python_executable,
            )
        )
        lines.extend(
            (
                f"## {suite.name}",
                f"- score key: {suite.score_key}",
                f"- goal: {suite.goal}",
                f"- command: `{command}`",
                "",
            )
        )
    return "\n".join(lines).strip() + "\n"


def render_codex_architecture_summary_markdown(
    environment: CodexEvalEnvironment,
) -> str:
    lines = [
        "# Architecture Summary",
        "",
        "## 1. 入口",
        "",
        "- 主な実行入口:",
        "  - `start_hub.py`",
        "  - `scripts/codex_usecase_environment.py`",
        "- 対象の初期読書順:",
    ]
    for entry in environment.onboarding:
        lines.append(f"  - `{entry.path}`: {entry.label}")

    lines.extend(
        (
            "",
            "## 2. 責務分割",
            "",
            "- state core: `inner_os/`",
            "- expression bridge: `inner_os/expression/`, `emot_terrain_lab/hub/runtime.py`",
            "- runtime / ops: `emot_terrain_lab/`, `ops/`",
            "- docs / evaluation: `docs/core/`, `inner_os/evaluation/`, `tests/`",
            "",
            "## 3. 主なデータフロー",
            "",
            "1. 入力 -> contact / event化",
            "2. shared field / readout -> boundary -> surface",
            "3. residual / thread / evaluation suite -> 次状態と改善ループ",
            "",
            "## 4. 変更候補ファイル",
            "",
            "- 最小変更で済む候補:",
            "  - `inner_os/expression/`",
            "  - `locales/ja.json`",
            "  - `emot_terrain_lab/hub/runtime.py`",
            "- 近傍で確認が必要な候補:",
            "  - `inner_os/evaluation/`",
            "  - `docs/core/`",
            "  - `tests/`",
            "",
            "## 5. 回帰観点",
            "",
            "- 既存テスト:",
        )
    )
    for suite in environment.suites:
        lines.append(f"  - `{suite.name}`: {', '.join(suite.test_targets)}")
    lines.extend(
        (
            "- 追加が必要な fixture:",
            "  - 変更対象の score key に対応する fixture を 1 本以上追加",
            "",
            "## 6. 不確実性",
            "",
            "- まだ読めていないもの:",
            "  - 対象機能の近傍 module 以外の詳細",
            "- 仮説に留まるもの:",
            "  - live 実走前の自然さ評価",
            "",
            f"> template source: `{environment.architecture_template_path}`",
        )
    )
    return "\n".join(lines).strip() + "\n"


def selected_codex_eval_suites(
    environment: CodexEvalEnvironment,
    names: Iterable[str] | None = None,
) -> tuple[CodexEvalSuite, ...]:
    requested = {str(item).strip().lower() for item in names or () if str(item).strip()}
    if not requested:
        return environment.suites
    return tuple(
        suite
        for suite in environment.suites
        if suite.name.lower() in requested
    )
