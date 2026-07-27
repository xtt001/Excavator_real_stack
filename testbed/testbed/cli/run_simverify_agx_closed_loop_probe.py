"""Run a bounded, non-promotable SimVerify checkpoint against live AGX."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from testbed.policies.offline_eval import load_policy_for_episode
from testbed.simverify.agx_closed_loop_probe import (
    ExternalAgxWorker,
    external_git_provenance,
    run_bounded_closed_loop_probe,
    validate_probe_bundle,
)
from testbed.simverify.contracts import git_provenance


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tb-run-simverify-agx-closed-loop-probe"
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--pact-root", type=Path, required=True)
    parser.add_argument("--unity-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--current-sector",
        choices=("left", "center", "right"),
        required=True,
    )
    parser.add_argument(
        "--next-sector",
        choices=("left", "center", "right"),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-ticks", type=int, default=10)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5057)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference-precision", default="fp32")
    parser.add_argument("--no-save-images", action="store_true")
    parser.add_argument("--allow-dirty-external", action="store_true")
    args = parser.parse_args()

    repository = args.repo_root.resolve(strict=True)
    current_git = git_provenance(repository)
    if (
        not current_git.get("git_available")
        or current_git.get("branch") != "v2.0.0-simVerify"
        or bool(current_git.get("dirty"))
    ):
        raise ValueError(
            "AGX probe requires a clean v2.0.0-simVerify worktree"
        )
    pact = external_git_provenance(args.pact_root)
    unity = external_git_provenance(args.unity_root)
    if (
        (pact["dirty"] or unity["dirty"])
        and not args.allow_dirty_external
    ):
        raise ValueError(
            "external PACT/Unity checkout is dirty; pass "
            "--allow-dirty-external only for explicitly non-promotable diagnostics"
        )
    bundle = validate_probe_bundle(args.bundle_root)
    policy = load_policy_for_episode(
        bundle_dir=args.bundle_root,
        ckpt_path=args.bundle_root / "policy_best.ckpt",
        resolved_config_path=None,
        stats_path=None,
        max_episode_len=8000,
        temporal_agg=True,
        device=args.device,
        inference_precision=args.inference_precision,
    )
    worker_script = repository / "scripts/simverify_agx_env_worker.py"
    environment = ExternalAgxWorker(
        repo_root=repository,
        pact_root=args.pact_root,
        worker_script=worker_script,
        python_executable=sys.executable,
        host=args.host,
        port=args.port,
        timeout_s=args.timeout,
    )
    try:
        result = run_bounded_closed_loop_probe(
            policy=policy,
            environment=environment,
            output_root=args.output_root,
            bundle_contract=bundle,
            current_git=current_git,
            external_provenance={"pact": pact, "unity": unity},
            current_sector=args.current_sector,
            next_sector=args.next_sector,
            seed=args.seed,
            policy_ticks=args.policy_ticks,
            save_images=not args.no_save_images,
        )
    finally:
        environment.close()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
