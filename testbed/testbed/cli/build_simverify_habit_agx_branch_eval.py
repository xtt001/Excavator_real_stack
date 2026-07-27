"""Build a paired gated-condition AGX branch diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from testbed.simverify.contracts import git_provenance
from testbed.simverify.habit_agx_branch_eval import (
    build_habit_agx_branch_diagnostic,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tb-build-simverify-habit-agx-branch-eval"
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--repeat-root", type=Path, required=True)
    parser.add_argument("--treatment-root", type=Path, required=True)
    parser.add_argument("--definition-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--terminal-window-policy-ticks", type=int, default=20)
    args = parser.parse_args()

    repository = args.repo_root.resolve(strict=True)
    current_git = git_provenance(repository)
    if (
        not current_git.get("git_available")
        or current_git.get("branch") != "v2.0.0-simVerify"
        or bool(current_git.get("dirty"))
    ):
        raise ValueError(
            "paired branch evaluation requires a clean v2.0.0-simVerify worktree"
        )
    result = build_habit_agx_branch_diagnostic(
        reference_root=args.reference_root,
        repeat_root=args.repeat_root,
        treatment_root=args.treatment_root,
        definition_root=args.definition_root,
        output_root=args.output_root,
        current_git=current_git,
        terminal_window_policy_ticks=args.terminal_window_policy_ticks,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
