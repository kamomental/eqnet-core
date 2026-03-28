from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inner_os.evaluation import (
    build_codex_eval_pytest_command,
    default_codex_eval_environment,
    render_codex_architecture_summary_markdown,
    render_codex_eval_plan_markdown,
    render_codex_onboarding_markdown,
    selected_codex_eval_suites,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Codex use-case onboarding / evaluation environment helper."
    )
    parser.add_argument(
        "mode",
        choices=("onboarding", "summary", "plan", "run"),
        help="出力モードを選ぶ",
    )
    parser.add_argument(
        "--suite",
        action="append",
        default=[],
        help="評価または表示する suite 名。未指定なら全 suite。",
    )
    parser.add_argument(
        "--python",
        dest="python_executable",
        default="python",
        help="pytest 実行に使う Python 実行ファイル。",
    )
    parser.add_argument(
        "--write",
        default="",
        help="Markdown を書き出すパス。未指定なら stdout。",
    )
    args = parser.parse_args(argv)

    environment = default_codex_eval_environment(REPO_ROOT)

    if args.mode == "onboarding":
        body = render_codex_onboarding_markdown(environment)
        _emit(body, args.write)
        return 0

    if args.mode == "summary":
        body = render_codex_architecture_summary_markdown(environment)
        _emit(body, args.write)
        return 0

    if args.mode == "plan":
        body = render_codex_eval_plan_markdown(
            environment,
            suite_names=args.suite,
            python_executable=args.python_executable,
        )
        _emit(body, args.write)
        return 0

    suites = selected_codex_eval_suites(environment, args.suite)
    if not suites:
        raise SystemExit("指定された suite が見つかりません。")

    report_lines = [
        "# Codex Evaluation Run",
        "",
        f"- repo root: `{environment.repo_root}`",
        "",
    ]
    overall_exit_code = 0
    for suite in suites:
        command = build_codex_eval_pytest_command(
            suite,
            python_executable=args.python_executable,
        )
        result = subprocess.run(
            command,
            cwd=environment.repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        report_lines.extend(
            (
                f"## {suite.name}",
                f"- score key: {suite.score_key}",
                f"- exit code: `{result.returncode}`",
                f"- command: `{' '.join(command)}`",
                "",
                "```text",
                (result.stdout or "").strip(),
                "```",
                "",
            )
        )
        if result.stderr:
            report_lines.extend(
                (
                    "```text",
                    result.stderr.strip(),
                    "```",
                    "",
                )
            )
        overall_exit_code = max(overall_exit_code, int(result.returncode))

    body = "\n".join(report_lines).strip() + "\n"
    _emit(body, args.write or environment.report_path)
    if not args.write:
        print(body)
    return overall_exit_code


def _emit(body: str, path: str) -> None:
    target = str(path or "").strip()
    if not target:
        print(body)
        return
    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(body, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
