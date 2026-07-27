"""Build a checksum-bound paired matrix from bounded AGX probe runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from testbed.simverify.agx_closed_loop_matrix import (
    build_closed_loop_paired_matrix,
)
from testbed.simverify.contracts import git_provenance


def main() -> None:
    parser = argparse.ArgumentParser(prog="tb-build-simverify-agx-closed-loop-matrix")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--m0-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--run-root",
        type=Path,
        action="append",
        required=True,
    )
    args = parser.parse_args()

    repository = args.repo_root.resolve(strict=True)
    current_git = git_provenance(repository)
    if (
        not current_git.get("git_available")
        or current_git.get("branch") != "v2.0.0-simVerify"
        or bool(current_git.get("dirty"))
    ):
        raise ValueError("paired matrix requires a clean v2.0.0-simVerify worktree")
    result = build_closed_loop_paired_matrix(
        run_roots=args.run_root,
        m0_root=args.m0_root,
        output_root=args.output_root,
        current_git=current_git,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
